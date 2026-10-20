use crate::model::{GraphNameRef, NamedOrBlankNodeRef, QuadRef, TermRef};
use crate::storage::CorruptionError;
pub use crate::storage::error::StorageError;
use crate::storage::numeric_encoder::{
    Decoder, EncodedQuad, EncodedTerm, StrHash, StrHashHasher, StrLookup, insert_term,
};
use dashmap::iter::Iter;
use dashmap::mapref::entry::Entry;
use dashmap::{DashMap, DashSet};
use oxrdf::Quad;
use rustc_hash::FxHasher;
use std::borrow::Borrow;
use std::hash::{BuildHasherDefault, Hash, Hasher};
use std::marker::PhantomData;
use std::mem::{take, transmute};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, RwLock, Weak};

/// In-memory storage working with MVCC
///
/// Each quad and graph name is annotated by a version range, allowing to read old versions while updates are applied.
/// To simplify the implementation, a single write transaction is currently allowed. This restriction should be lifted in the future.
#[derive(Clone)]
pub struct MemoryStorage {
    state: Arc<RwLock<Arc<State>>>,
    version_counter: Arc<AtomicUsize>,
    transaction_counter: Arc<AtomicUsize>,
    transaction_lock: Arc<Lock>,
}

/// The quads, named graphs and strings of the store.
///
/// [`MemoryStorage::compact`] replaces the whole state, so readers and transactions hold the
/// state they started on.
#[derive(Default)]
struct State {
    content: Content,
    id2str: DashMap<StrHash, String, BuildHasherDefault<StrHashHasher>>,
}

#[derive(Default)]
struct Content {
    quad_set: DashSet<Arc<QuadListNode>, BuildHasherDefault<FxHasher>>,
    last_quad: RwLock<Option<Weak<QuadListNode>>>,
    last_quad_by_subject:
        DashMap<EncodedTerm, (Weak<QuadListNode>, u64), BuildHasherDefault<FxHasher>>,
    last_quad_by_predicate:
        DashMap<EncodedTerm, (Weak<QuadListNode>, u64), BuildHasherDefault<FxHasher>>,
    last_quad_by_object:
        DashMap<EncodedTerm, (Weak<QuadListNode>, u64), BuildHasherDefault<FxHasher>>,
    last_quad_by_graph_name:
        DashMap<EncodedTerm, (Weak<QuadListNode>, u64), BuildHasherDefault<FxHasher>>,
    graphs: DashMap<EncodedTerm, VersionRange>,
}

impl MemoryStorage {
    pub fn new() -> Self {
        Self {
            state: Arc::default(),
            version_counter: Arc::new(AtomicUsize::new(0)),
            transaction_counter: Arc::new(AtomicUsize::new(usize::MAX >> 1)),
            transaction_lock: Arc::new(Lock::new()),
        }
    }

    pub fn snapshot(&self) -> MemoryStorageReader<'static> {
        // The version is read while holding the state: `compact` swaps the state under the write
        // lock, so a snapshot's version is always one its state contains.
        let state = self.state.read().unwrap();
        MemoryStorageReader {
            state: Arc::clone(&state),
            snapshot_id: self.version_counter.load(Ordering::Acquire),
            _lifetime: PhantomData,
        }
    }

    pub fn start_transaction(&self) -> MemoryStorageTransaction<'_> {
        // We ensure there is only one transaction running
        let transaction_guard = self.transaction_lock.lock();
        let transaction_id = self.transaction_counter.fetch_add(1, Ordering::Acquire);
        let snapshot_id = self.version_counter.load(Ordering::Relaxed);
        MemoryStorageTransaction {
            storage: self,
            // `compact` takes the transaction lock, so the state cannot change under us.
            state: Arc::clone(&self.state.read().unwrap()),
            log: Vec::new(),
            transaction_id,
            snapshot_id,
            _transaction_guard: transaction_guard,
            committed: false,
        }
    }

    pub fn bulk_loader(&self) -> MemoryStorageBulkLoader<'_> {
        MemoryStorageBulkLoader {
            transaction: self.start_transaction(),
            done: 0,
            hooks: Vec::new(),
        }
    }

    /// Rebuilds the state from the quads and named graphs visible now.
    ///
    /// Removing a quad only closes its version range, so removed quads, rolled-back insertions
    /// and their strings otherwise stay in memory and in the lists scans walk. The rebuild keeps
    /// the live ones only, in their insertion order.
    ///
    /// Writers wait while it runs; readers do not. A snapshot taken before keeps reading the
    /// state it started on, which is freed when the last such snapshot is dropped.
    #[expect(clippy::unwrap_in_result)]
    pub fn compact(&self) -> Result<(), StorageError> {
        let _transaction_guard = self.transaction_lock.lock();
        let reader = self.snapshot();
        let compacted = State::default();

        let mut live = Vec::new();
        let mut next = reader.state.content.last_quad.read().unwrap().clone();
        while let Some(current) = next.take().and_then(|c| c.upgrade()) {
            if reader.is_node_in_range(&current) {
                live.push(current.quad.clone());
            }
            next.clone_from(&current.previous);
        }
        // The chain runs from the newest quad back, so push the oldest first.
        for encoded in live.into_iter().rev() {
            let quad = reader.decode_quad(&encoded)?;
            compacted.insert_term(quad.subject.as_ref().into(), &encoded.subject);
            compacted.insert_term(quad.predicate.as_ref().into(), &encoded.predicate);
            compacted.insert_term(quad.object.as_ref(), &encoded.object);
            // Every version a reader of the compacted state can hold is at least the current one.
            compacted.content.push(encoded, VersionRange::Start(0));
        }
        for entry in &reader.state.content.graphs {
            if reader.is_in_range(entry.value()) {
                let graph_name = reader.decode_named_or_blank_node(entry.key())?;
                compacted.insert_term(graph_name.as_ref().into(), entry.key());
                compacted
                    .content
                    .graphs
                    .insert(entry.key().clone(), VersionRange::Start(0));
            }
        }

        *self.state.write().unwrap() = Arc::new(compacted);
        Ok(())
    }
}

impl State {
    fn insert_term(&self, term: TermRef<'_>, encoded: &EncodedTerm) {
        insert_term(term, encoded, &mut |key, value| self.insert_str(key, value))
    }

    fn insert_str(&self, key: &StrHash, value: &str) {
        let inserted = self.id2str.entry(*key).or_insert_with(|| value.into());
        debug_assert_eq!(*inserted, value, "Hash conflict for two strings");
    }
}

impl Content {
    /// Adds a node for a quad not yet in the set, at the head of all its lists.
    fn push(&self, quad: EncodedQuad, range: VersionRange) -> Arc<QuadListNode> {
        let node = Arc::new(QuadListNode {
            range: Mutex::new(range),
            previous: self.last_quad.read().unwrap().clone(),
            previous_subject: self
                .last_quad_by_subject
                .view(&quad.subject, |_, (node, _)| Weak::clone(node)),
            previous_predicate: self
                .last_quad_by_predicate
                .view(&quad.predicate, |_, (node, _)| Weak::clone(node)),
            previous_object: self
                .last_quad_by_object
                .view(&quad.object, |_, (node, _)| Weak::clone(node)),
            previous_graph_name: self
                .last_quad_by_graph_name
                .view(&quad.graph_name, |_, (node, _)| Weak::clone(node)),
            quad,
        });
        self.quad_set.insert(Arc::clone(&node));
        *self.last_quad.write().unwrap() = Some(Arc::downgrade(&node));
        for (list, term) in [
            (&self.last_quad_by_subject, &node.quad.subject),
            (&self.last_quad_by_predicate, &node.quad.predicate),
            (&self.last_quad_by_object, &node.quad.object),
            (&self.last_quad_by_graph_name, &node.quad.graph_name),
        ] {
            list.entry(term.clone())
                .and_modify(|(e, count)| {
                    *e = Arc::downgrade(&node);
                    *count += 1;
                })
                .or_insert_with(|| (Arc::downgrade(&node), 1));
        }
        node
    }
}

#[derive(Clone)]
#[must_use]
pub struct MemoryStorageReader<'a> {
    state: Arc<State>,
    snapshot_id: usize,
    _lifetime: PhantomData<&'a ()>,
}

impl<'a> MemoryStorageReader<'a> {
    pub fn len(&self) -> usize {
        self.state
            .content
            .quad_set
            .iter()
            .filter(|e| self.is_node_in_range(e))
            .count()
    }

    pub fn is_empty(&self) -> bool {
        !self
            .state
            .content
            .quad_set
            .iter()
            .any(|e| self.is_node_in_range(&e))
    }

    pub fn contains(&self, quad: &EncodedQuad) -> bool {
        self.state
            .content
            .quad_set
            .get(quad)
            .is_some_and(|node| self.is_node_in_range(&node))
    }

    pub fn quads_for_pattern(
        &self,
        subject: Option<&EncodedTerm>,
        predicate: Option<&EncodedTerm>,
        object: Option<&EncodedTerm>,
        graph_name: Option<&EncodedTerm>,
    ) -> QuadIterator<'a> {
        fn get_start_and_count(
            map: &DashMap<EncodedTerm, (Weak<QuadListNode>, u64), BuildHasherDefault<FxHasher>>,
            term: Option<&EncodedTerm>,
        ) -> (Option<Weak<QuadListNode>>, u64) {
            let Some(term) = term else {
                return (None, u64::MAX);
            };
            map.view(term, |_, (node, count)| (Some(Weak::clone(node)), *count))
                .unwrap_or_default()
        }

        let (subject_start, subject_count) =
            get_start_and_count(&self.state.content.last_quad_by_subject, subject);
        let (predicate_start, predicate_count) =
            get_start_and_count(&self.state.content.last_quad_by_predicate, predicate);
        let (object_start, object_count) =
            get_start_and_count(&self.state.content.last_quad_by_object, object);
        let (graph_name_start, graph_name_count) =
            get_start_and_count(&self.state.content.last_quad_by_graph_name, graph_name);

        let (start, kind) = if subject.is_some()
            && subject_count <= predicate_count
            && subject_count <= object_count
            && subject_count <= graph_name_count
        {
            (subject_start, QuadIteratorKind::Subject)
        } else if predicate.is_some()
            && predicate_count <= object_count
            && predicate_count <= graph_name_count
        {
            (predicate_start, QuadIteratorKind::Predicate)
        } else if object.is_some() && object_count <= graph_name_count {
            (object_start, QuadIteratorKind::Object)
        } else if graph_name.is_some() {
            (graph_name_start, QuadIteratorKind::GraphName)
        } else {
            (
                self.state.content.last_quad.read().unwrap().clone(),
                QuadIteratorKind::All,
            )
        };
        QuadIterator {
            reader: self.clone(),
            current: start,
            kind,
            expect_subject: if kind == QuadIteratorKind::Subject {
                None
            } else {
                subject.cloned()
            },
            expect_predicate: if kind == QuadIteratorKind::Predicate {
                None
            } else {
                predicate.cloned()
            },
            expect_object: if kind == QuadIteratorKind::Object {
                None
            } else {
                object.cloned()
            },
            expect_graph_name: if kind == QuadIteratorKind::GraphName {
                None
            } else {
                graph_name.cloned()
            },
        }
    }

    #[expect(unsafe_code)]
    pub fn named_graphs(&self) -> MemoryDecodingGraphIterator<'a> {
        MemoryDecodingGraphIterator {
            reader: self.clone(),
            // SAFETY: this is fine, the owning struct also owns the iterated data structure
            iter: unsafe {
                transmute::<Iter<'_, _, _>, Iter<'a, _, _>>(self.state.content.graphs.iter())
            },
        }
    }

    pub fn contains_named_graph(&self, graph_name: &EncodedTerm) -> bool {
        self.state
            .content
            .graphs
            .get(graph_name)
            .is_some_and(|range| self.is_in_range(&range))
    }

    pub fn contains_str(&self, key: &StrHash) -> bool {
        self.state.id2str.contains_key(key)
    }

    /// Validate that all the storage invariants held in the data
    #[expect(clippy::unwrap_in_result)]
    pub fn validate(&self) -> Result<(), StorageError> {
        // All used named graphs are in graph set
        let expected_quad_len = self.state.content.quad_set.len() as u64;

        // last quad chain
        let mut next = self.state.content.last_quad.read().unwrap().clone();
        let mut count_last_quad = 0;
        while let Some(current) = next.take().and_then(|c| c.upgrade()) {
            count_last_quad += 1;
            if !self
                .state
                .content
                .quad_set
                .get(&current.quad)
                .is_some_and(|e| Arc::ptr_eq(&e, &current))
            {
                return Err(
                    CorruptionError::new("Quad in previous chain but not in quad set").into(),
                );
            }
            self.decode_quad(&current.quad)?;
            if !current.quad.graph_name.is_default_graph()
                && !self
                    .state
                    .content
                    .graphs
                    .contains_key(&current.quad.graph_name)
            {
                return Err(
                    CorruptionError::new("Quad in named graph that does not exists").into(),
                );
            }
            next.clone_from(&current.previous);
        }
        if count_last_quad != expected_quad_len {
            return Err(CorruptionError::new("Too many quads in quad_set").into());
        }

        // By subject chain
        let mut count_last_by_subject = 0;
        for entry in &self.state.content.last_quad_by_subject {
            let mut next = Some(Weak::clone(&entry.value().0));
            let mut element_count = 0;
            while let Some(current) = next.take().and_then(|n| n.upgrade()) {
                element_count += 1;
                if current.quad.subject != *entry.key() {
                    return Err(CorruptionError::new("Quad in wrong list").into());
                }
                if !self
                    .state
                    .content
                    .quad_set
                    .get(&current.quad)
                    .is_some_and(|e| Arc::ptr_eq(&e, &current))
                {
                    return Err(
                        CorruptionError::new("Quad in previous chain but not in quad set").into(),
                    );
                }
                next.clone_from(&current.previous_subject);
            }
            if element_count != entry.value().1 {
                return Err(CorruptionError::new("Too many quads in a chain").into());
            }
            count_last_by_subject += element_count;
        }
        if count_last_by_subject != expected_quad_len {
            return Err(CorruptionError::new("Too many quads in quad_set").into());
        }

        // By predicate chains
        let mut count_last_by_predicate = 0;
        for entry in &self.state.content.last_quad_by_predicate {
            let mut next = Some(Weak::clone(&entry.value().0));
            let mut element_count = 0;
            while let Some(current) = next.take().and_then(|n| n.upgrade()) {
                element_count += 1;
                if current.quad.predicate != *entry.key() {
                    return Err(CorruptionError::new("Quad in wrong list").into());
                }
                if !self
                    .state
                    .content
                    .quad_set
                    .get(&current.quad)
                    .is_some_and(|e| Arc::ptr_eq(&e, &current))
                {
                    return Err(
                        CorruptionError::new("Quad in previous chain but not in quad set").into(),
                    );
                }
                next.clone_from(&current.previous_predicate);
            }
            if element_count != entry.value().1 {
                return Err(CorruptionError::new("Too many quads in a chain").into());
            }
            count_last_by_predicate += element_count;
        }
        if count_last_by_predicate != expected_quad_len {
            return Err(CorruptionError::new("Too many quads in quad_set").into());
        }

        // By object chains
        let mut count_last_by_object = 0;
        for entry in &self.state.content.last_quad_by_object {
            let mut next = Some(Weak::clone(&entry.value().0));
            let mut element_count = 0;
            while let Some(current) = next.take().and_then(|n| n.upgrade()) {
                element_count += 1;
                if current.quad.object != *entry.key() {
                    return Err(CorruptionError::new("Quad in wrong list").into());
                }
                if !self
                    .state
                    .content
                    .quad_set
                    .get(&current.quad)
                    .is_some_and(|e| Arc::ptr_eq(&e, &current))
                {
                    return Err(
                        CorruptionError::new("Quad in previous chain but not in quad set").into(),
                    );
                }
                next.clone_from(&current.previous_object);
            }
            if element_count != entry.value().1 {
                return Err(CorruptionError::new("Too many quads in a chain").into());
            }
            count_last_by_object += element_count;
        }
        if count_last_by_object != expected_quad_len {
            return Err(CorruptionError::new("Too many quads in quad_set").into());
        }

        // By graph_name chains
        let mut count_last_by_graph_name = 0;
        for entry in &self.state.content.last_quad_by_graph_name {
            let mut next = Some(Weak::clone(&entry.value().0));
            let mut element_count = 0;
            while let Some(current) = next.take().and_then(|n| n.upgrade()) {
                element_count += 1;
                if current.quad.graph_name != *entry.key() {
                    return Err(CorruptionError::new("Quad in wrong list").into());
                }
                if !self
                    .state
                    .content
                    .quad_set
                    .get(&current.quad)
                    .is_some_and(|e| Arc::ptr_eq(&e, &current))
                {
                    return Err(
                        CorruptionError::new("Quad in previous chain but not in quad set").into(),
                    );
                }
                next.clone_from(&current.previous_graph_name);
            }
            if element_count != entry.value().1 {
                return Err(CorruptionError::new("Too many quads in a chain").into());
            }
            count_last_by_graph_name += element_count;
        }
        if count_last_by_graph_name != expected_quad_len {
            return Err(CorruptionError::new("Too many quads in quad_set").into());
        }

        Ok(())
    }

    fn is_in_range(&self, range: &VersionRange) -> bool {
        range.contains(self.snapshot_id)
    }

    fn is_node_in_range(&self, node: &QuadListNode) -> bool {
        let range = node.range.lock().unwrap();
        self.is_in_range(&range)
    }
}

impl StrLookup for MemoryStorageReader<'_> {
    fn get_str(&self, key: &StrHash) -> Result<Option<String>, StorageError> {
        Ok(self.state.id2str.view(key, |_, v| v.clone()))
    }
}

#[must_use]
pub struct MemoryStorageTransaction<'a> {
    storage: &'a MemoryStorage,
    state: Arc<State>,
    log: Vec<LogEntry>,
    transaction_id: usize,
    snapshot_id: usize,
    committed: bool,
    _transaction_guard: LockGuard<'a>,
}

impl MemoryStorageTransaction<'_> {
    pub fn reader(&self) -> MemoryStorageReader<'_> {
        MemoryStorageReader {
            state: Arc::clone(&self.state),
            snapshot_id: self.transaction_id,
            _lifetime: PhantomData,
        }
    }

    pub fn insert(&mut self, quad: QuadRef<'_>) {
        let encoded: EncodedQuad = quad.into();
        if let Some(node) = self
            .state
            .content
            .quad_set
            .get(&encoded)
            .map(|node| Arc::clone(&node))
        {
            let added = node.range.lock().unwrap().add(self.transaction_id);
            if added {
                self.log.push(LogEntry::QuadNode(node));
                if !quad.graph_name.is_default_graph()
                    && self
                        .state
                        .content
                        .graphs
                        .get_mut(&encoded.graph_name)
                        .unwrap()
                        .add(self.transaction_id)
                {
                    self.log.push(LogEntry::Graph(encoded.graph_name.clone()));
                }
            }
        } else {
            let node = self
                .state
                .content
                .push(encoded.clone(), VersionRange::Start(self.transaction_id));
            self.state
                .insert_term(quad.subject.into(), &encoded.subject);
            self.state
                .insert_term(quad.predicate.into(), &encoded.predicate);
            self.state.insert_term(quad.object, &encoded.object);

            match quad.graph_name {
                GraphNameRef::NamedNode(graph_name) => {
                    self.insert_encoded_named_graph(graph_name.into(), encoded.graph_name.clone());
                }
                GraphNameRef::BlankNode(graph_name) => {
                    self.insert_encoded_named_graph(graph_name.into(), encoded.graph_name.clone());
                }
                GraphNameRef::DefaultGraph => (),
            }
            self.log.push(LogEntry::QuadNode(node));
        }
    }

    pub fn insert_named_graph(&mut self, graph_name: NamedOrBlankNodeRef<'_>) {
        self.insert_encoded_named_graph(graph_name, graph_name.into())
    }

    fn insert_encoded_named_graph(
        &mut self,
        graph_name: NamedOrBlankNodeRef<'_>,
        encoded_graph_name: EncodedTerm,
    ) {
        let added = match self.state.content.graphs.entry(encoded_graph_name.clone()) {
            Entry::Occupied(mut entry) => entry.get_mut().add(self.transaction_id),
            Entry::Vacant(entry) => {
                entry.insert(VersionRange::Start(self.transaction_id));
                self.state
                    .insert_term(graph_name.into(), &encoded_graph_name);
                true
            }
        };
        if added {
            self.log.push(LogEntry::Graph(encoded_graph_name));
        }
    }

    pub fn remove(&mut self, quad: QuadRef<'_>) {
        self.remove_encoded(&quad.into())
    }

    fn remove_encoded(&mut self, quad: &EncodedQuad) {
        let Some(node) = self
            .state
            .content
            .quad_set
            .get(quad)
            .map(|node| Arc::clone(&node))
        else {
            return;
        };
        let removed = node.range.lock().unwrap().remove(self.transaction_id);
        if removed {
            self.log.push(LogEntry::QuadNode(node));
        }
    }

    pub fn clear_graph(&mut self, graph_name: GraphNameRef<'_>) {
        self.clear_encoded_graph(&graph_name.into())
    }

    fn clear_encoded_graph(&mut self, graph_name: &EncodedTerm) {
        let mut next = self
            .state
            .content
            .last_quad_by_graph_name
            .view(graph_name, |_, (node, _)| Weak::clone(node));
        while let Some(current) = next.take().and_then(|c| c.upgrade()) {
            if current.range.lock().unwrap().remove(self.transaction_id) {
                self.log.push(LogEntry::QuadNode(Arc::clone(&current)));
            }
            next.clone_from(&current.previous_graph_name);
        }
    }

    pub fn clear_all_named_graphs(&mut self) {
        let graph_names = self.reader().named_graphs().collect::<Vec<_>>();
        for graph_name in graph_names {
            self.clear_encoded_graph(&graph_name)
        }
    }

    pub fn clear_all_graphs(&mut self) {
        self.state.content.quad_set.iter().for_each(|node| {
            if node.range.lock().unwrap().remove(self.transaction_id) {
                self.log.push(LogEntry::QuadNode(Arc::clone(&node)));
            }
        });
    }

    pub fn remove_named_graph(&mut self, graph_name: NamedOrBlankNodeRef<'_>) {
        self.remove_encoded_named_graph(&graph_name.into())
    }

    fn remove_encoded_named_graph(&mut self, graph_name: &EncodedTerm) {
        self.clear_encoded_graph(graph_name);
        let removed = self
            .state
            .content
            .graphs
            .get_mut(graph_name)
            .is_some_and(|mut entry| entry.value_mut().remove(self.transaction_id));
        if removed {
            self.log.push(LogEntry::Graph(graph_name.clone()));
        }
    }

    pub fn remove_all_named_graphs(&mut self) {
        self.clear_all_named_graphs();
        self.do_remove_graphs();
    }

    fn do_remove_graphs(&mut self) {
        self.state.content.graphs.iter_mut().for_each(|mut entry| {
            if entry.value_mut().remove(self.transaction_id) {
                self.log.push(LogEntry::Graph(entry.key().clone()));
            }
        });
    }

    pub fn clear(&mut self) {
        self.clear_all_graphs();
        self.do_remove_graphs();
    }

    pub fn commit(mut self) {
        let new_version_id = self.snapshot_id + 1;
        for operation in take(&mut self.log) {
            match operation {
                LogEntry::QuadNode(node) => {
                    node.range
                        .lock()
                        .unwrap()
                        .upgrade_transaction(self.transaction_id, new_version_id);
                }
                LogEntry::Graph(graph_name) => {
                    if let Some(mut entry) = self.state.content.graphs.get_mut(&graph_name) {
                        entry
                            .value_mut()
                            .upgrade_transaction(self.transaction_id, new_version_id)
                    }
                }
            }
        }
        self.storage
            .version_counter
            .store(new_version_id, Ordering::Release);
        self.committed = true;
    }
}

impl Drop for MemoryStorageTransaction<'_> {
    fn drop(&mut self) {
        // We roll back
        if !self.committed {
            for operation in take(&mut self.log) {
                match operation {
                    LogEntry::QuadNode(node) => {
                        node.range
                            .lock()
                            .unwrap()
                            .rollback_transaction(self.transaction_id);
                    }
                    LogEntry::Graph(graph_name) => {
                        if let Some(mut entry) = self.state.content.graphs.get_mut(&graph_name) {
                            entry.value_mut().rollback_transaction(self.transaction_id)
                        }
                    }
                }
            }
            // Nodes of rolled-back insertions stay until `MemoryStorage::compact`.
        }
    }
}

#[must_use]
pub struct QuadIterator<'a> {
    reader: MemoryStorageReader<'a>,
    current: Option<Weak<QuadListNode>>,
    kind: QuadIteratorKind,
    expect_subject: Option<EncodedTerm>,
    expect_predicate: Option<EncodedTerm>,
    expect_object: Option<EncodedTerm>,
    expect_graph_name: Option<EncodedTerm>,
}

#[derive(PartialEq, Eq, Clone, Copy)]
enum QuadIteratorKind {
    All,
    Subject,
    Predicate,
    Object,
    GraphName,
}

impl Iterator for QuadIterator<'_> {
    type Item = EncodedQuad;

    fn next(&mut self) -> Option<EncodedQuad> {
        loop {
            let current = self.current.take()?.upgrade()?;
            self.current = match self.kind {
                QuadIteratorKind::All => current.previous.clone(),
                QuadIteratorKind::Subject => current.previous_subject.clone(),
                QuadIteratorKind::Predicate => current.previous_predicate.clone(),
                QuadIteratorKind::Object => current.previous_object.clone(),
                QuadIteratorKind::GraphName => current.previous_graph_name.clone(),
            };
            if !self.reader.is_node_in_range(&current) {
                continue;
            }
            if let Some(expect_subject) = &self.expect_subject {
                if current.quad.subject != *expect_subject {
                    continue;
                }
            }
            if let Some(expect_predicate) = &self.expect_predicate {
                if current.quad.predicate != *expect_predicate {
                    continue;
                }
            }
            if let Some(expect_object) = &self.expect_object {
                if current.quad.object != *expect_object {
                    continue;
                }
            }
            if let Some(expect_graph_name) = &self.expect_graph_name {
                if current.quad.graph_name != *expect_graph_name {
                    continue;
                }
            }
            return Some(current.quad.clone());
        }
    }
}

#[must_use]
pub struct MemoryDecodingGraphIterator<'a> {
    reader: MemoryStorageReader<'a>, // Needed to make sure the underlying map is not GCed
    iter: Iter<'a, EncodedTerm, VersionRange>,
}

impl Iterator for MemoryDecodingGraphIterator<'_> {
    type Item = EncodedTerm;

    fn next(&mut self) -> Option<EncodedTerm> {
        loop {
            let entry = self.iter.next()?;
            if self.reader.is_in_range(entry.value()) {
                return Some(entry.key().clone());
            }
        }
    }
}

#[must_use]
pub struct MemoryStorageBulkLoader<'a> {
    transaction: MemoryStorageTransaction<'a>,
    done: u64,
    hooks: Vec<Box<dyn Fn(u64) + Send + Sync>>,
}

impl MemoryStorageBulkLoader<'_> {
    pub fn on_progress(mut self, callback: impl Fn(u64) + Send + Sync + 'static) -> Self {
        self.hooks.push(Box::new(callback));
        self
    }

    pub fn load_batch(&mut self, new_quads: Vec<Quad>) {
        for quad in new_quads {
            self.transaction.insert(quad.as_ref());
            self.done += 1;
            if self.done.is_multiple_of(1_000_000) {
                for hook in &self.hooks {
                    hook(self.done);
                }
            }
        }
    }

    pub fn commit(self) {
        self.transaction.commit();
    }
}

enum LogEntry {
    QuadNode(Arc<QuadListNode>),
    Graph(EncodedTerm),
}

struct QuadListNode {
    quad: EncodedQuad,
    range: Mutex<VersionRange>,
    previous: Option<Weak<Self>>,
    previous_subject: Option<Weak<Self>>,
    previous_predicate: Option<Weak<Self>>,
    previous_object: Option<Weak<Self>>,
    previous_graph_name: Option<Weak<Self>>,
}

impl PartialEq for QuadListNode {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.quad == other.quad
    }
}

impl Eq for QuadListNode {}

impl Hash for QuadListNode {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.quad.hash(state)
    }
}

impl Borrow<EncodedQuad> for Arc<QuadListNode> {
    fn borrow(&self) -> &EncodedQuad {
        &self.quad
    }
}

// TODO: reduce the size to 128bits
#[derive(Default, Eq, PartialEq, Clone)]
enum VersionRange {
    #[default]
    Empty,
    Start(usize),
    StartEnd(usize, usize),
    Bigger(Box<[usize]>),
}

impl VersionRange {
    fn contains(&self, version: usize) -> bool {
        match self {
            VersionRange::Empty => false,
            VersionRange::Start(start) => *start <= version,
            VersionRange::StartEnd(start, end) => *start <= version && version < *end,
            VersionRange::Bigger(range) => {
                for start_end in range.chunks(2) {
                    match start_end {
                        [start, end] if *start <= version && version < *end => {
                            return true;
                        }
                        [start] if *start <= version => {
                            return true;
                        }
                        _ => (),
                    }
                }
                false
            }
        }
    }

    fn add(&mut self, version: usize) -> bool {
        match self {
            VersionRange::Empty => {
                *self = VersionRange::Start(version);
                true
            }
            VersionRange::Start(_) => false,
            VersionRange::StartEnd(start, end) => {
                *self = if version == *end {
                    VersionRange::Start(*start)
                } else {
                    VersionRange::Bigger(Box::new([*start, *end, version]))
                };
                true
            }
            VersionRange::Bigger(vec) => {
                if vec.len() % 2 == 0 {
                    *self = VersionRange::Bigger(if vec.ends_with(&[version]) {
                        pop_boxed_slice(vec)
                    } else {
                        push_boxed_slice(vec, version)
                    });
                    true
                } else {
                    false
                }
            }
        }
    }

    fn remove(&mut self, version: usize) -> bool {
        match self {
            VersionRange::Empty | VersionRange::StartEnd(_, _) => false,
            VersionRange::Start(start) => {
                *self = if *start == version {
                    VersionRange::Empty
                } else {
                    VersionRange::StartEnd(*start, version)
                };
                true
            }
            VersionRange::Bigger(vec) => {
                if vec.len() % 2 == 0 {
                    false
                } else {
                    *self = if vec.ends_with(&[version]) {
                        match vec.as_ref() {
                            [start, end, _] => Self::StartEnd(*start, *end),
                            _ => Self::Bigger(pop_boxed_slice(vec)),
                        }
                    } else {
                        Self::Bigger(push_boxed_slice(vec, version))
                    };
                    true
                }
            }
        }
    }

    fn upgrade_transaction(&mut self, transaction_id: usize, version_id: usize) {
        match self {
            VersionRange::Empty => (),
            VersionRange::Start(start) => {
                if *start == transaction_id {
                    *start = version_id;
                }
            }
            VersionRange::StartEnd(_, end) => {
                if *end == transaction_id {
                    *end = version_id
                }
            }
            VersionRange::Bigger(vec) => {
                if vec.ends_with(&[transaction_id]) {
                    vec[vec.len() - 1] = version_id
                }
            }
        }
    }

    fn rollback_transaction(&mut self, transaction_id: usize) {
        match self {
            VersionRange::Empty => (),
            VersionRange::Start(start) => {
                if *start == transaction_id {
                    *self = VersionRange::Empty;
                }
            }
            VersionRange::StartEnd(start, end) => {
                if *end == transaction_id {
                    *self = VersionRange::Start(*start)
                }
            }
            VersionRange::Bigger(vec) => {
                if vec.ends_with(&[transaction_id]) {
                    *self = match vec.as_ref() {
                        [start, end, _] => Self::StartEnd(*start, *end),
                        _ => Self::Bigger(pop_boxed_slice(vec)),
                    }
                }
            }
        }
    }
}

fn push_boxed_slice<T: Copy>(slice: &[T], element: T) -> Box<[T]> {
    let mut out = Vec::with_capacity(slice.len() + 1);
    out.extend_from_slice(slice);
    out.push(element);
    out.into_boxed_slice()
}

fn pop_boxed_slice<T: Copy>(slice: &[T]) -> Box<[T]> {
    slice[..slice.len() - 1].into()
}

struct Lock {
    mutex: Mutex<bool>,
    condvar: Condvar,
}

impl Lock {
    fn new() -> Self {
        Self {
            mutex: Mutex::new(false),
            condvar: Condvar::new(),
        }
    }

    fn lock(&self) -> LockGuard<'_> {
        *self
            .condvar
            .wait_while(self.mutex.lock().unwrap(), |v| *v)
            .unwrap() = true;
        LockGuard {
            mutex: &self.mutex,
            condvar: &self.condvar,
        }
    }
}

struct LockGuard<'a> {
    mutex: &'a Mutex<bool>,
    condvar: &'a Condvar,
}

impl Drop for LockGuard<'_> {
    fn drop(&mut self) {
        *self.mutex.lock().unwrap() = false;
        self.condvar.notify_one();
    }
}

#[cfg(test)]
#[expect(clippy::panic_in_result_fn)]
mod tests {
    use super::*;
    use oxrdf::{Literal, NamedNodeRef};

    #[test]
    fn test_range() {
        let mut range = VersionRange::default();

        assert!(range.add(1));
        assert!(!range.add(1));
        assert!(range.contains(1));
        assert!(!range.contains(0));
        assert!(range.contains(2));

        assert!(range.remove(1));
        assert!(!range.remove(1));
        assert!(!range.contains(1));

        assert!(range.add(1));
        assert!(range.remove(2));
        assert!(!range.remove(2));
        assert!(range.contains(1));
        assert!(!range.contains(2));

        assert!(range.add(2));
        assert!(range.contains(3));

        assert!(range.remove(2));
        assert!(range.add(4));
        assert!(range.remove(6));
        assert!(!range.contains(3));
        assert!(range.contains(4));
        assert!(!range.contains(6));
    }

    #[test]
    fn test_upgrade() {
        let mut range = VersionRange::default();

        assert!(range.add(1000));
        range.upgrade_transaction(999, 1);
        assert!(!range.contains(1));
        range.upgrade_transaction(1000, 1);
        assert!(range.contains(1));

        assert!(range.remove(1000));
        range.upgrade_transaction(999, 2);
        assert!(range.contains(2));
        range.upgrade_transaction(1000, 2);
        assert!(!range.contains(2));

        assert!(range.add(1000));
        range.upgrade_transaction(999, 3);
        assert!(!range.contains(3));
        range.upgrade_transaction(1000, 3);
        assert!(range.contains(3));
    }

    #[test]
    fn test_rollback() {
        let mut range = VersionRange::default();

        assert!(range.add(1000));
        range.rollback_transaction(999);
        assert!(range.contains(1000));
        range.rollback_transaction(1000);
        assert!(!range.contains(1));
    }

    #[test]
    fn test_transaction() -> Result<(), StorageError> {
        let example = NamedNodeRef::new_unchecked("http://example.com/1");
        let example2 = NamedNodeRef::new_unchecked("http://example.com/2");
        let encoded_example = EncodedTerm::from(example);
        let encoded_example2 = EncodedTerm::from(example2);
        let default_quad = QuadRef::new(example, example, example, GraphNameRef::DefaultGraph);
        let encoded_default_quad = EncodedQuad::from(default_quad);
        let named_graph_quad = QuadRef::new(example, example, example, example);
        let encoded_named_graph_quad = EncodedQuad::from(named_graph_quad);

        let storage = MemoryStorage::new();

        // We start with a graph
        let snapshot = storage.snapshot();
        let mut transaction = storage.start_transaction();
        transaction.insert_named_graph(example.into());
        transaction.commit();
        assert!(!snapshot.contains_named_graph(&encoded_example));
        assert!(storage.snapshot().contains_named_graph(&encoded_example));
        storage.snapshot().validate()?;

        // We add two quads
        let snapshot = storage.snapshot();
        let mut transaction = storage.start_transaction();
        transaction.insert(default_quad);
        transaction.insert(named_graph_quad);
        transaction.commit();
        assert!(!snapshot.contains(&encoded_default_quad));
        assert!(!snapshot.contains(&encoded_named_graph_quad));
        assert!(storage.snapshot().contains(&encoded_default_quad));
        assert!(storage.snapshot().contains(&encoded_named_graph_quad));
        storage.snapshot().validate()?;

        // We remove the quads
        let snapshot = storage.snapshot();
        let mut transaction = storage.start_transaction();
        transaction.remove(default_quad);
        transaction.remove_named_graph(example.into());
        transaction.commit();
        assert!(snapshot.contains(&encoded_default_quad));
        assert!(snapshot.contains(&encoded_named_graph_quad));
        assert!(snapshot.contains_named_graph(&encoded_example));
        assert!(!storage.snapshot().contains(&encoded_default_quad));
        assert!(!storage.snapshot().contains(&encoded_named_graph_quad));
        assert!(!storage.snapshot().contains_named_graph(&encoded_example));
        storage.snapshot().validate()?;

        // We add the quads again but rollback
        let snapshot = storage.snapshot();
        let mut transaction = storage.start_transaction();
        transaction.insert(default_quad);
        transaction.insert(named_graph_quad);
        transaction.insert_named_graph(example2.into());
        drop(transaction);
        assert!(!snapshot.contains(&encoded_default_quad));
        assert!(!snapshot.contains(&encoded_named_graph_quad));
        assert!(!snapshot.contains_named_graph(&encoded_example));
        assert!(!snapshot.contains_named_graph(&encoded_example2));
        assert!(!storage.snapshot().contains(&encoded_default_quad));
        assert!(!storage.snapshot().contains(&encoded_named_graph_quad));
        assert!(!storage.snapshot().contains_named_graph(&encoded_example));
        assert!(!storage.snapshot().contains_named_graph(&encoded_example2));
        storage.snapshot().validate()?;

        // We add quads and graph, then clear
        storage.bulk_loader().load_batch(vec![
            default_quad.into_owned(),
            named_graph_quad.into_owned(),
        ]);
        let mut transaction = storage.start_transaction();
        transaction.insert_named_graph(example2.into());
        transaction.commit();
        let mut transaction = storage.start_transaction();
        transaction.clear();
        transaction.commit();
        assert!(!storage.snapshot().contains(&encoded_default_quad));
        assert!(!storage.snapshot().contains(&encoded_named_graph_quad));
        assert!(!storage.snapshot().contains_named_graph(&encoded_example));
        assert!(!storage.snapshot().contains_named_graph(&encoded_example2));
        assert!(storage.snapshot().is_empty());
        storage.snapshot().validate()?;

        Ok(())
    }

    #[test]
    fn test_compact() -> Result<(), StorageError> {
        const SUBJECT: NamedNodeRef<'_> = NamedNodeRef::new_unchecked("http://example.com/s");
        const PREDICATE: NamedNodeRef<'_> = NamedNodeRef::new_unchecked("http://example.com/p");
        const GRAPH: NamedNodeRef<'_> = NamedNodeRef::new_unchecked("http://example.com/g");
        // Long enough to be stored in the string table rather than inlined.
        fn value(i: usize) -> Literal {
            Literal::new_simple_literal(format!("value number {i} of the stream"))
        }
        fn quad(literal: &Literal) -> QuadRef<'_> {
            QuadRef::new(SUBJECT, PREDICATE, literal, GRAPH)
        }
        let (subject, predicate, graph) = (SUBJECT, PREDICATE, GRAPH);

        let storage = MemoryStorage::new();
        for i in 0..100 {
            let mut transaction = storage.start_transaction();
            if i > 0 {
                transaction.remove(quad(&value(i - 1)));
            }
            transaction.insert(quad(&value(i)));
            transaction.commit();
        }
        let mut transaction = storage.start_transaction();
        transaction.insert(QuadRef::new(
            subject,
            predicate,
            &value(1000),
            GraphNameRef::DefaultGraph,
        ));
        drop(transaction);

        let before = storage.snapshot();
        let open_iterator = before.quads_for_pattern(Some(&subject.into()), None, None, None);
        assert_eq!(before.state.content.quad_set.len(), 101);

        storage.compact()?;

        let after = storage.snapshot();
        after.validate()?;
        assert_eq!(after.state.content.quad_set.len(), 1);
        // The three IRIs and the current value.
        assert_eq!(after.state.id2str.len(), 4);
        assert!(after.contains(&quad(&value(99)).into()));
        assert!(after.contains_named_graph(&graph.into()));

        // Readers that started before keep their own state.
        assert_eq!(
            open_iterator.collect::<Vec<_>>(),
            vec![EncodedQuad::from(quad(&value(99)))]
        );
        assert_eq!(before.len(), 1);
        before.validate()?;

        // Writes after a compaction see the compacted state.
        let mut transaction = storage.start_transaction();
        transaction.remove(quad(&value(99)));
        transaction.commit();
        assert!(storage.snapshot().is_empty());
        assert!(after.contains(&quad(&value(99)).into()));
        storage.snapshot().validate()?;
        Ok(())
    }
}

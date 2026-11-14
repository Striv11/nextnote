// Tiny IndexedDB wrapper: one slot holding the session JSON string. Browser-only (no Node tests).
const DB = "nextnote", STORE = "sessions", KEY = "autosave";
function open() {
  return new Promise((res, rej) => {
    const r = indexedDB.open(DB, 1);
    r.onupgradeneeded = () => r.result.createObjectStore(STORE);
    r.onsuccess = () => res(r.result);
    r.onerror = () => rej(r.error);
  });
}
async function tx(mode, fn) {
  const db = await open();
  return new Promise((res, rej) => {
    const t = db.transaction(STORE, mode), req = fn(t.objectStore(STORE));
    t.oncomplete = () => { db.close(); res(req.result); };
    t.onerror = t.onabort = () => { db.close(); rej(t.error); };
  });
}
export const saveSession = (text) => tx("readwrite", (s) => s.put(text, KEY));
export const loadSession = () => tx("readonly", (s) => s.get(KEY));

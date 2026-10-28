const ACCOUNTS_KEY = 'intent-annotator:accounts:v1';
const SESSION_KEY = 'intent-annotator:session:v1';
const PBKDF2_ITERATIONS = 100000;

export function normalizeEmail(email) {
  return String(email || '').trim().toLowerCase();
}

export function isValidEmail(email) {
  const value = normalizeEmail(email);
  // Practical client-side check; not a full RFC validator.
  return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value) && value.length <= 254;
}

export function isValidPassword(password) {
  return typeof password === 'string' && password.length >= 8 && password.length <= 128;
}

function bytesToBase64(bytes) {
  let binary = '';
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary);
}

function base64ToBytes(value) {
  const binary = atob(value);
  return Uint8Array.from(binary, char => char.charCodeAt(0));
}

export async function hashPassword(password, saltBytes) {
  const material = await crypto.subtle.importKey('raw', new TextEncoder().encode(password), 'PBKDF2', false, ['deriveBits']);
  const bits = await crypto.subtle.deriveBits(
    {name: 'PBKDF2', salt: saltBytes, iterations: PBKDF2_ITERATIONS, hash: 'SHA-256'},
    material,
    256
  );
  return bytesToBase64(new Uint8Array(bits));
}

function readAccounts() {
  try {
    const raw = localStorage.getItem(ACCOUNTS_KEY);
    if (!raw) return {schema_version: 1, accounts: []};
    const parsed = JSON.parse(raw);
    if (parsed?.schema_version !== 1 || !Array.isArray(parsed.accounts)) throw new Error('Unrecognized account store');
    return {
      schema_version: 1,
      accounts: parsed.accounts.filter(account =>
        typeof account?.email === 'string' &&
        typeof account?.salt === 'string' &&
        typeof account?.hash === 'string'
      )
    };
  } catch {
    throw new Error('Saved accounts could not be read from this browser.');
  }
}

function writeAccounts(store) {
  localStorage.setItem(ACCOUNTS_KEY, JSON.stringify(store));
}

export function getSession() {
  try {
    const raw = localStorage.getItem(SESSION_KEY);
    if (!raw) return null;
    const session = JSON.parse(raw);
    if (typeof session?.email !== 'string' || !isValidEmail(session.email)) return null;
    return {email: normalizeEmail(session.email), signed_in_at: typeof session.signed_in_at === 'string' ? session.signed_in_at : null};
  } catch {
    return null;
  }
}

export function setSession(email) {
  const normalized = normalizeEmail(email);
  localStorage.setItem(SESSION_KEY, JSON.stringify({email: normalized, signed_in_at: new Date().toISOString()}));
  return getSession();
}

export function clearSession() {
  localStorage.removeItem(SESSION_KEY);
}

export function responseStorageKey(email) {
  return `intent-annotator:responses:v1:${normalizeEmail(email)}`;
}

export async function createAccount(email, password) {
  if (!isValidEmail(email)) throw new Error('Enter a valid email address.');
  if (!isValidPassword(password)) throw new Error('Password must be 8–128 characters.');
  const store = readAccounts();
  const normalized = normalizeEmail(email);
  if (store.accounts.some(account => account.email === normalized)) {
    throw new Error('An account with this email already exists. Sign in instead.');
  }
  const salt = crypto.getRandomValues(new Uint8Array(16));
  const hash = await hashPassword(password, salt);
  store.accounts.push({
    email: normalized,
    salt: bytesToBase64(salt),
    hash,
    created_at: new Date().toISOString()
  });
  writeAccounts(store);
  return setSession(normalized);
}

export async function signIn(email, password) {
  if (!isValidEmail(email)) throw new Error('Enter a valid email address.');
  if (typeof password !== 'string' || !password) throw new Error('Enter your password.');
  const store = readAccounts();
  const normalized = normalizeEmail(email);
  const account = store.accounts.find(item => item.email === normalized);
  if (!account) throw new Error('No account found for that email. Create an account first.');
  const hash = await hashPassword(password, base64ToBytes(account.salt));
  if (hash !== account.hash) throw new Error('Incorrect email or password.');
  return setSession(normalized);
}

export function signOut() {
  clearSession();
}

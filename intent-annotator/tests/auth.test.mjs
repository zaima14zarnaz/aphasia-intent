import test from 'node:test';
import assert from 'node:assert/strict';
import {normalizeEmail, isValidEmail, isValidPassword, hashPassword, createAccount, signIn, signOut, getSession, responseStorageKey} from '../assets/auth.mjs';

const memory = new Map();
globalThis.localStorage = {
  getItem: key => (memory.has(key) ? memory.get(key) : null),
  setItem: (key, value) => { memory.set(key, String(value)); },
  removeItem: key => { memory.delete(key); },
  clear: () => { memory.clear(); }
};

test('email and password validation', () => {
  assert.equal(normalizeEmail('  SLP@Clinic.EDU '), 'slp@clinic.edu');
  assert.equal(isValidEmail('slp@clinic.edu'), true);
  assert.equal(isValidEmail('not-an-email'), false);
  assert.equal(isValidPassword('short'), false);
  assert.equal(isValidPassword('long-enough'), true);
});

test('password hashing is deterministic for a salt', async () => {
  const salt = new Uint8Array(16).fill(7);
  const first = await hashPassword('annotator-pass', salt);
  const second = await hashPassword('annotator-pass', salt);
  const other = await hashPassword('different-pass', salt);
  assert.equal(first, second);
  assert.notEqual(first, other);
});

test('create account, sign in, and scope response keys', async () => {
  memory.clear();
  const session = await createAccount('slp01@clinic.edu', 'securepass');
  assert.equal(session.email, 'slp01@clinic.edu');
  assert.equal(getSession().email, 'slp01@clinic.edu');
  assert.equal(responseStorageKey(session.email), 'intent-annotator:responses:v1:slp01@clinic.edu');
  await assert.rejects(() => createAccount('SLP01@clinic.edu', 'securepass'), /already exists/);
  signOut();
  assert.equal(getSession(), null);
  const again = await signIn('slp01@clinic.edu', 'securepass');
  assert.equal(again.email, 'slp01@clinic.edu');
  await assert.rejects(() => signIn('slp01@clinic.edu', 'wrong-password'), /Incorrect/);
});

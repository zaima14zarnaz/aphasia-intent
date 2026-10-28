import {FIELDS, ROLES, EXAMPLES, emptyAnnotation, hasRequired, cleanAnnotation, toCSV, parseImport} from './model.mjs';
import {createAccount, signIn, signOut, getSession, responseStorageKey, isValidEmail, isValidPassword} from './auth.mjs';

const $ = id => document.getElementById(id);
let clips = [], index = 0, examples = {...EXAMPLES};
let responses = new Map(), storageBlocked = false;
let storageMessage = '';
let session = null;
let authMode = 'create';
let workspaceReady = false;
const video = $('video');

function notice(message) { $('notice').textContent = message; $('notice').hidden = !message; }
function current() { return clips[index]; }
function annotation(id = current()?.id) { return responses.get(id) || emptyAnnotation(); }
function storageKey() { return responseStorageKey(session.email); }
function annotatorId() { return session?.email || ''; }

function showAuthError(message) {
  $('auth-error').textContent = message;
  $('auth-error').hidden = !message;
}

function setAuthMode(mode) {
  authMode = mode;
  const creating = mode === 'create';
  $('auth-tab-create').setAttribute('aria-selected', String(creating));
  $('auth-tab-signin').setAttribute('aria-selected', String(!creating));
  $('auth-panel').setAttribute('aria-labelledby', creating ? 'auth-tab-create' : 'auth-tab-signin');
  $('auth-title').textContent = creating ? 'Create an account to begin' : 'Sign in to continue';
  $('auth-submit').textContent = creating ? 'Create account' : 'Sign in';
  $('auth-confirm-field').hidden = !creating;
  $('auth-password').autocomplete = creating ? 'new-password' : 'current-password';
  $('auth-password-confirm').required = creating;
  $('auth-password-help').textContent = creating
    ? 'At least 8 characters. Accounts are stored only in this browser.'
    : 'Use the password you created on this device.';
  document.querySelector('.skip-link').setAttribute('href', '#auth-email');
  showAuthError('');
}

function showAuthScreen() {
  $('auth-screen').hidden = false;
  $('app-shell').hidden = true;
  setAuthMode(authMode);
  $('auth-email').focus();
}

function showWorkspace() {
  $('auth-screen').hidden = true;
  $('app-shell').hidden = false;
  $('account-email').textContent = session.email;
  document.querySelector('#app-shell .skip-link')?.focus({preventScroll: true});
}

function restore() {
  responses = new Map();
  storageBlocked = false;
  storageMessage = '';
  document.body.classList.remove('storage-error');
  try {
    let raw = localStorage.getItem(storageKey());
    // Adopt pre-login browser saves once, then keep them under this account.
    if (!raw) {
      const legacy = localStorage.getItem('intent-annotator:responses:v1');
      if (legacy) {
        raw = legacy;
        localStorage.setItem(storageKey(), legacy);
      }
    }
    if (!raw) {
      $('save-status').textContent = 'Responses save in this browser';
      return;
    }
    const state = JSON.parse(raw);
    if (state.schema_version !== 1 || !Array.isArray(state.responses)) throw new Error('Unrecognized saved data');
    responses = new Map(state.responses.filter(item => typeof item?.id === 'string').map(item => [item.id, cleanAnnotation(item.annotation)]));
    if (responses.size) $('save-status').textContent = 'Saved responses restored';
    else $('save-status').textContent = 'Responses save in this browser';
  } catch {
    storageBlocked = true;
    storageMessage = 'Browser saving is unavailable or saved data could not be read. Existing storage has been preserved. Export JSON to keep your work.';
    $('save-status').textContent = 'Browser saving unavailable — export a backup';
    document.body.classList.add('storage-error');
  }
}

function persist() {
  if (!session) return false;
  if (storageBlocked) { notice(storageMessage); return false; }
  try {
    localStorage.setItem(storageKey(), JSON.stringify({
      schema_version: 1,
      annotator_id: annotatorId(),
      responses: [...responses].map(([id, annotation]) => ({id, annotation}))
    }));
    $('save-status').textContent = 'All changes saved in this browser';
    document.body.classList.remove('storage-error');
    return true;
  } catch {
    $('save-status').textContent = 'Not saved — export a JSON backup';
    document.body.classList.add('storage-error');
    notice('Browser storage is full or unavailable. Your current responses remain in this tab. Export JSON before closing it.');
    return false;
  }
}

function writeExamples() {
  for (const key of FIELDS) {
    const text = (examples[key] || '').trim();
    $(key).placeholder = text ? `Example: ${text}` : '';
  }
}

function buildRoles() {
  if ($('role-options').childElementCount) return;
  for (const [id, text] of ROLES) {
    const label = document.createElement('label');
    const input = document.createElement('input');
    input.type = 'checkbox'; input.name = 'roles'; input.value = id;
    input.addEventListener('change', () => {
      const checks = [...$('role-options').querySelectorAll('input')];
      if (input.checked && id === 'unclear') checks.forEach(check => { if (check !== input) check.checked = false; });
      else if (input.checked) checks.find(check => check.value === 'unclear').checked = false;
      saveForm();
    });
    label.append(input, document.createTextNode(text));
    $('role-options').append(label);
  }
}

function buildTabs() {
  $('clip-tabs').replaceChildren();
  clips.forEach((clip, i) => {
    const button = document.createElement('button');
    button.id = 'clip-tab-' + i; button.type = 'button'; button.role = 'tab';
    button.setAttribute('aria-controls', 'clip-panel');
    const dot = document.createElement('span'); dot.className = 'tab-dot'; dot.setAttribute('aria-hidden', 'true');
    const title = document.createElement('span'); title.className = 'tab-title'; title.textContent = clip.title || clip.filename || 'Clip ' + (i + 1);
    button.append(dot, title); button.title = clip.filename || clip.title;
    button.addEventListener('click', () => selectClip(i));
    button.addEventListener('keydown', event => {
      let target;
      if (event.key === 'ArrowRight') target = (i + 1) % clips.length;
      if (event.key === 'ArrowLeft') target = (i - 1 + clips.length) % clips.length;
      if (event.key === 'Home') target = 0;
      if (event.key === 'End') target = clips.length - 1;
      if (target !== undefined) { event.preventDefault(); selectClip(target, true); }
    });
    $('clip-tabs').append(button);
  });
}

function updateProgress() {
  const total = clips.length;
  const count = clips.filter(clip => annotation(clip.id).completed).length;
  $('progress-text').textContent = `${count} of ${total} clips complete`;
  const percent = total ? Math.round(count / total * 100) : 0;
  $('progress-percent').textContent = percent + '%';
  $('progress').max = total || 1; $('progress').value = count;
  $('progress').textContent = percent + '%';
  clips.forEach((clip, i) => {
    const button = $('clip-tab-' + i), saved = annotation(clip.id);
    if (!button) return;
    button.setAttribute('aria-selected', String(i === index)); button.tabIndex = i === index ? 0 : -1;
    button.classList.toggle('tab-complete', saved.completed);
    button.querySelector('.tab-dot').textContent = saved.completed ? '✓' : (i + 1);
    button.setAttribute('aria-label', `${clip.title || clip.filename}, ${saved.completed ? 'complete' : 'incomplete'}`);
  });
  if (!current()) return;
  const saved = annotation();
  const started = FIELDS.some(key => saved[key].trim()) || saved.roles.length > 0;
  $('clip-state').textContent = saved.completed ? 'Complete' : started ? 'In progress' : 'Not started';
  $('clip-state').classList.toggle('complete', saved.completed);
  $('complete').textContent = saved.completed ? 'Clip complete ✓' : 'Mark clip complete ✓';
  $('complete').disabled = saved.completed;
  $('completion-help').textContent = saved.completed ? 'This clip is complete. Editing a response reopens it for review.' : 'Placeholder text is guidance only and is not saved as a response.';
}

function saveForm() {
  if (!current()) return;
  const saved = annotation();
  for (const key of FIELDS) saved[key] = $(key).value;
  saved.roles = [...$('role-options').querySelectorAll('input:checked')].map(input => input.value);
  saved.completed = false; saved.updated_at = new Date().toISOString();
  responses.set(current().id, saved); persist(); updateProgress();
}

function showVideoMessage(title, text) {
  $('video-message-title').textContent = title;
  $('video-message-text').textContent = text;
  $('video-message').hidden = false;
}

function selectClip(target, focus = false) {
  if (target < 0 || target >= clips.length) return;
  video.pause(); index = target;
  const clip = current(), saved = annotation();
  $('clip-panel').hidden = false;
  $('clip-panel').setAttribute('aria-labelledby', 'clip-tab-' + index);
  $('clip-title').textContent = clip.title || clip.filename;
  $('clip-task').textContent = 'Task: ' + (clip.task || 'Sandwich task');
  $('transcript').textContent = clip.transcript || 'Placeholder transcript — replace with the written transcript.';
  $('placeholder-tag').hidden = clip.placeholder_context === false;
  for (const key of FIELDS) $(key).value = saved[key];
  for (const input of $('role-options').querySelectorAll('input')) input.checked = saved.roles.includes(input.value);
  $('video-message').hidden = true;
  video.removeAttribute('src');
  if (clip.src) { video.src = clip.src; $('replay').disabled = false; }
  else { showVideoMessage('Video unavailable', 'This clip has no playable video source.'); $('replay').disabled = true; }
  video.load(); video.playbackRate = Number($('playback-speed').value);
  $('clip-position').textContent = `Clip ${index + 1} of ${clips.length}`;
  $('bottom-position').textContent = `${index + 1} / ${clips.length}`;
  $('previous').disabled = index === 0; $('next').disabled = index === clips.length - 1;
  updateProgress();
  const tab = $('clip-tab-' + index);
  if (focus) tab.focus();
  const tabs = $('clip-tabs');
  if (tab.offsetLeft < tabs.scrollLeft || tab.offsetLeft + tab.offsetWidth > tabs.scrollLeft + tabs.clientWidth) tabs.scrollLeft = tab.offsetLeft - tabs.offsetLeft;
}

function setClips(next) { clips = next; index = 0; buildTabs(); if (clips.length) selectClip(0); else { $('clip-panel').hidden = true; updateProgress(); } }

function normalizeManifest(manifest) {
  if (!Array.isArray(manifest.clips)) throw new Error('data/clips.json must contain a clips array.');
  const ids = new Set();
  return manifest.clips.map((clip, i) => {
    if (!clip || typeof clip.id !== 'string' || !clip.id || ids.has(clip.id)) throw new Error('Each clip needs a unique nonempty string id.');
    ids.add(clip.id);
    let src = '';
    if (typeof clip.src === 'string' && clip.src) {
      const url = new URL(clip.src, document.baseURI);
      if (!['http:', 'https:'].includes(url.protocol)) throw new Error('Video sources must use relative paths or HTTP(S) URLs.');
      if (location.protocol === 'https:' && url.protocol === 'http:') throw new Error('Use HTTPS video URLs on an HTTPS site.');
      src = clip.src;
    }
    return {...clip, src, title: typeof clip.title === 'string' ? clip.title : `Clip ${i + 1}`, filename: typeof clip.filename === 'string' ? clip.filename : clip.id};
  });
}

function exportPayload() {
  return {app: 'Intent Annotator', schema_version: 1, annotator_id: annotatorId(), exported_at: new Date().toISOString(), annotations: clips.map(clip => ({clip: {id: clip.id, filename: clip.filename, src: clip.src.startsWith('blob:') ? '' : clip.src, task: clip.task || '', transcript: clip.transcript || ''}, annotation: annotation(clip.id)}))};
}

function download(content, extension, mime) {
  if (!clips.length) { notice('Load a clip set before exporting.'); return; }
  const blobURL = URL.createObjectURL(new Blob([content], {type: mime}));
  const anchor = document.createElement('a');
  const annotator = annotatorId().replace(/[^a-zA-Z0-9_-]+/g, '_') || 'annotator';
  anchor.href = blobURL; anchor.download = `intent-annotations_${annotator}_${new Date().toISOString().replaceAll(':', '-').slice(0, 19)}.${extension}`;
  document.body.append(anchor); anchor.click(); anchor.remove();
  setTimeout(() => URL.revokeObjectURL(blobURL), 1000);
  document.querySelector('.export-menu').open = false;
  notice('Export downloaded. Keep the JSON file as a backup; send your exported annotations to the study coordinator.');
}

async function loadClips() {
  try {
    const response = await fetch('data/clips.json', {cache: 'no-store'});
    if (!response.ok) throw new Error(`Clip list returned HTTP ${response.status}.`);
    const manifest = await response.json();
    if (manifest.examples && typeof manifest.examples === 'object') for (const key of FIELDS) if (typeof manifest.examples[key] === 'string') examples[key] = manifest.examples[key];
    writeExamples(); setClips(normalizeManifest(manifest));
    if (!clips.length) notice('No clips are configured in data/clips.json.');
    else if (clips.every(clip => !clip.src)) notice('Clips are listed, but none have a playable video source yet.');
    if (storageBlocked) notice(storageMessage);
  } catch (error) {
    notice('Could not load the clip list: ' + error.message + ' Run this site through an HTTP server, rather than opening index.html directly.');
  }
}

function bindWorkspace() {
  if (workspaceReady) return;
  workspaceReady = true;
  buildRoles();
  writeExamples();
  for (const key of FIELDS) $(key).addEventListener('input', saveForm);
  $('import-button').addEventListener('click', () => $('import-json').click());
  $('previous').addEventListener('click', () => selectClip(index - 1));
  $('next').addEventListener('click', () => selectClip(index + 1));
  $('playback-speed').addEventListener('change', () => { video.playbackRate = Number($('playback-speed').value); });
  $('replay').addEventListener('click', () => { video.currentTime = 0; video.play().catch(() => notice('Playback could not start. Try the video controls or check that the file is supported.')); });
  video.addEventListener('error', () => {
    if (current()?.src) showVideoMessage('This video could not be loaded', 'Check that the clip is available and uses a browser-supported MP4 codec.');
  });
  video.addEventListener('loadedmetadata', () => { $('video-message').hidden = true; video.playbackRate = Number($('playback-speed').value); });
  $('annotation-form').addEventListener('submit', event => {
    event.preventDefault();
    if (!hasRequired(annotation())) {
      notice('Complete Movement description, Gesture meaning, Intent, and Reasoning before marking this clip complete.');
      const missing = FIELDS.find(key => !$(key).value.trim()); if (missing) $(missing).focus(); return;
    }
    const saved = annotation(); saved.completed = true; saved.updated_at = new Date().toISOString(); responses.set(current().id, saved);
    const stored = persist(); updateProgress();
    if (stored && clips.every(clip => annotation(clip.id).completed)) {
      notice('All clips are complete. Export your annotations from “Annotation files” above.');
    } else {
      notice('');
    }
  });

  $('export-json').addEventListener('click', () => download(JSON.stringify(exportPayload(), null, 2), 'json', 'application/json'));
  $('export-csv').addEventListener('click', () => { const payload = exportPayload(); download(toCSV(payload.annotations, payload.annotator_id), 'csv', 'text/csv;charset=utf-8'); });
  $('import-json').addEventListener('change', async event => {
    const file = event.target.files[0]; if (!file) return;
    try {
      const parsed = parseImport(JSON.parse(await file.text()), clips.map(clip => clip.id));
      const hasExisting = [...parsed.records.keys()].some(id => { const saved = annotation(id); return FIELDS.some(key => saved[key].trim()) || saved.roles.length; });
      if (hasExisting && !confirm('Importing will replace responses for matching clips in this browser. Export a backup first if you want to keep them. Continue?')) return;
      for (const [id, value] of parsed.records) responses.set(id, value);
      const stored = persist(); selectClip(index);
      notice(`Imported ${parsed.records.size} matching clip annotations.${parsed.ignored ? ` Ignored ${parsed.ignored} records from other clip sets.` : ''}${stored ? '' : ' Browser saving is unavailable; export JSON before closing this tab.'}`);
    } catch (error) { notice('Import failed: ' + error.message); }
    finally { event.target.value = ''; }
  });
  window.addEventListener('storage', event => {
    if (!session) return;
    if (event.key === storageKey()) {
      storageBlocked = true;
      storageMessage = 'Responses changed in another tab. Export a JSON backup from this tab, then reload to use the other tab’s saved responses. Browser saving is paused to avoid overwriting them.';
      $('save-status').textContent = 'Changed in another tab — export and reload';
      document.body.classList.add('storage-error');
      notice(storageMessage);
    }
  });
  $('sign-out').addEventListener('click', () => {
    if (responses.size && !confirm('Sign out? Your responses remain saved for this email on this device. Export a backup if you need a portable copy.')) return;
    signOut();
    session = null;
    clips = [];
    responses = new Map();
    notice('');
    showAuthScreen();
  });
}

async function enterWorkspace(nextSession) {
  session = nextSession;
  bindWorkspace();
  restore();
  showWorkspace();
  await loadClips();
}

$('auth-tab-create').addEventListener('click', () => setAuthMode('create'));
$('auth-tab-signin').addEventListener('click', () => setAuthMode('signin'));
$('auth-form').addEventListener('submit', async event => {
  event.preventDefault();
  showAuthError('');
  const email = $('auth-email').value;
  const password = $('auth-password').value;
  const confirm = $('auth-password-confirm').value;
  if (!isValidEmail(email)) { showAuthError('Enter a valid email address.'); $('auth-email').focus(); return; }
  if (authMode === 'create') {
    if (!isValidPassword(password)) { showAuthError('Password must be 8–128 characters.'); $('auth-password').focus(); return; }
    if (password !== confirm) { showAuthError('Passwords do not match.'); $('auth-password-confirm').focus(); return; }
  } else if (!password) {
    showAuthError('Enter your password.');
    $('auth-password').focus();
    return;
  }
  $('auth-submit').disabled = true;
  try {
    const next = authMode === 'create' ? await createAccount(email, password) : await signIn(email, password);
    $('auth-form').reset();
    await enterWorkspace(next);
  } catch (error) {
    showAuthError(error.message || 'Could not complete sign-in.');
  } finally {
    $('auth-submit').disabled = false;
  }
});

const existing = getSession();
if (existing) {
  $('auth-screen').hidden = true;
  enterWorkspace(existing);
} else {
  showAuthScreen();
}

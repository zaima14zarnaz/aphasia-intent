export const FIELDS = ['movement', 'meaning', 'intent', 'reasoning'];
export const ROLES = [
  ['add_information', 'Add information missing from speech.'],
  ['resolve_ambiguity', 'Resolve an ambiguous word.'],
  ['reinforce_transcript', 'Reinforce information already available in transcript.'],
  ['little_information', 'Provide little useful information.'],
  ['unclear', 'Unclear']
];
export const EXAMPLES = {
  movement: 'The participant raises the right hand, opens the palm upward, and moves it forward while tilting the head slightly to the left.',
  meaning: 'The open palm and forward movement depict offering an object to another person.',
  intent: 'The participant is describing giving an object to someone.',
  reasoning: 'The forward hand movement suggests transfer, while the spoken phrase “I gave it to her” supports the interpretation of giving.',
  role: 'Reinforce information already available in transcript.'
};
export function emptyAnnotation() {
  return {movement:'', meaning:'', intent:'', reasoning:'', roles:[], completed:false, updated_at:null};
}
export function hasRequired(annotation) {
  return FIELDS.every(key => typeof annotation?.[key] === 'string' && annotation[key].trim().length > 0);
}
export function cleanAnnotation(raw) {
  const result = emptyAnnotation();
  if (!raw || typeof raw !== 'object') return result;
  for (const key of FIELDS) result[key] = typeof raw[key] === 'string' ? raw[key] : '';
  result.roles = [...new Set(Array.isArray(raw.roles) ? raw.roles.filter(role => ROLES.some(([id]) => id === role)) : [])];
  if (result.roles.includes('unclear')) result.roles = ['unclear'];
  result.completed = raw.completed === true && hasRequired(result);
  result.updated_at = typeof raw.updated_at === 'string' ? raw.updated_at : null;
  return result;
}
export function csvCell(value) {
  let text = String(value ?? '');
  // Prevent spreadsheet software interpreting user-provided text as formulas.
  if (/^[\s]*[=+@-]/.test(text)) text = "'" + text;
  return '"' + text.replaceAll('"', '""') + '"';
}
export function toCSV(records, annotator) {
  const headers = ['annotator_id','clip_id','filename','video_url','task','transcript',...FIELDS,'roles','completed','updated_at'];
  const rows = records.map(record => [annotator,record.clip.id,record.clip.filename,record.clip.src,record.clip.task,record.clip.transcript,...FIELDS.map(key => record.annotation[key]),record.annotation.roles.map(id => ROLES.find(([key]) => key === id)?.[1] || id).join(' | '),record.annotation.completed,record.annotation.updated_at]);
  return '\uFEFF' + [headers,...rows].map(row => row.map(csvCell).join(',')).join('\r\n');
}
export function parseImport(payload, clipIds) {
  if (payload?.app !== 'Intent Annotator' || payload.schema_version !== 1 || !Array.isArray(payload.annotations)) throw new Error('Choose a JSON backup exported by Intent Annotator (schema version 1).');
  const validIds = new Set(clipIds), records = new Map();
  let ignored = 0;
  for (const record of payload.annotations) {
    if (typeof record?.clip?.id !== 'string' || !validIds.has(record.clip.id)) { ignored++; continue; }
    if (!record.annotation || typeof record.annotation !== 'object') throw new Error('The backup contains an invalid annotation. No responses were imported.');
    if (records.has(record.clip.id)) throw new Error('The backup contains duplicate clip IDs. No responses were imported.');
    records.set(record.clip.id,cleanAnnotation(record.annotation));
  }
  if (!records.size) throw new Error('No clip IDs in this backup match the currently loaded clips. Load the correct clip set first.');
  return {records,ignored,annotator:typeof payload.annotator_id === 'string' ? payload.annotator_id : ''};
}

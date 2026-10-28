import test from 'node:test';
import assert from 'node:assert/strict';
import {emptyAnnotation,hasRequired,cleanAnnotation,toCSV,parseImport} from '../assets/model.mjs';

const filled = {movement:'Hand rises',meaning:'Giving',intent:'Describe transfer',reasoning:'Speech and gesture agree',roles:['reinforce_transcript'],completed:true};
test('completion needs all four genuine responses; whitespace and examples do not count',() => {
  assert.equal(hasRequired(emptyAnnotation()),false);
  assert.equal(hasRequired(filled),true);
  assert.equal(cleanAnnotation({...filled,reasoning:'  '}).completed,false);
});
test('import validates clip identity and required fields before restoring completion',() => {
  const payload = {app:'Intent Annotator',schema_version:1,annotator_id:'SLP-01',annotations:[{clip:{id:'clip1.mp4'},annotation:{...filled,reasoning:'',roles:['unclear','unknown','reinforce_transcript']}},{clip:{id:'other.mp4'},annotation:filled}]};
  const parsed = parseImport(payload,['clip1.mp4']);
  assert.equal(parsed.ignored,1);
  assert.equal(parsed.records.get('clip1.mp4').completed,false);
  assert.deepEqual(parsed.records.get('clip1.mp4').roles,['unclear']);
  assert.throws(() => parseImport(payload,['unmatched.mp4']),/No clip IDs/);
  assert.throws(() => parseImport({...payload,annotations:[payload.annotations[0],payload.annotations[0]]},['clip1.mp4']),/duplicate/);
});
test('CSV preserves multiline text and quotes, and neutralizes spreadsheet formulas',() => {
  const csv = toCSV([{clip:{id:'clip1.mp4',filename:'clip1.mp4',src:'videos/clip1.mp4',task:'Task',transcript:'First line\nSecond, "quoted" line'},annotation:{...filled,movement:'=HYPERLINK("test")'}}],'SLP-01');
  assert.ok(csv.startsWith('\uFEFF'));
  assert.ok(csv.includes('First line\nSecond, ""quoted"" line'));
  assert.ok(csv.includes('"\'=HYPERLINK(""test"")"'));
});

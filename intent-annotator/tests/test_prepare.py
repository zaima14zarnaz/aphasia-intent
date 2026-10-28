import importlib.util
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

spec = importlib.util.spec_from_file_location('prepare_clips', Path(__file__).resolve().parents[1] / 'scripts/prepare_clips.py')
prepare_clips = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare_clips)

class PreparationTests(unittest.TestCase):
    def test_copy_selected_nested_video_and_encode_url(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source'
            (source / 'nested').mkdir(parents=True)
            (source / 'nested/clip #1.mp4').write_bytes(b'selected video fixture')
            (source / 'unselected.mp4').write_bytes(b'not selected')
            selection = root / 'selection.txt'
            selection.write_text('nested/clip #1.mp4\n')
            project = root / 'site'
            with patch.object(prepare_clips, 'ROOT', project):
                clips = prepare_clips.prepare(source,project / 'data/clips.json',selection)
            self.assertEqual(clips[0]['id'],'nested/clip #1.mp4')
            self.assertEqual(clips[0]['src'],'videos/nested/clip%20%231.mp4')
            self.assertEqual((project / 'videos/nested/clip #1.mp4').read_bytes(),b'selected video fixture')
            self.assertFalse((project / 'videos/unselected.mp4').exists())

    def test_selection_encoding_and_metadata_preservation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source'
            source.mkdir()
            for name in ['clip 2.mp4','clip10.MP4','clip1.mp4']:
                (source / name).write_bytes(b'test fixture')
            manifest = root / 'clips.json'
            manifest.write_text(json.dumps({'clips':[{'id':'clip 2.mp4','transcript':'Edited transcript','task':'Edited task','placeholder_context':False}],'examples':{'intent':'Custom example'}}))
            clips = prepare_clips.prepare(source,manifest,base_url='https://example.org/media')
            self.assertEqual([clip['id'] for clip in clips],['clip1.mp4','clip10.MP4','clip 2.mp4'])
            self.assertEqual(clips[2]['src'],'https://example.org/media/clip%202.mp4')
            self.assertEqual(clips[2]['transcript'],'Edited transcript')
            self.assertFalse(clips[2]['placeholder_context'])
            self.assertEqual(json.loads(manifest.read_text())['examples']['intent'],'Custom example')
            selection = root / 'selection.txt'
            selection.write_text('clip10.MP4\nclip1.mp4\n')
            self.assertEqual([clip['id'] for clip in prepare_clips.prepare(source,manifest,selection,manifest_only=True)],['clip10.MP4','clip1.mp4'])
            before = manifest.read_bytes()
            selection.write_text('../outside.mp4\n')
            with self.assertRaises(ValueError):
                prepare_clips.prepare(source,manifest,selection,manifest_only=True)
            self.assertEqual(manifest.read_bytes(),before)

if __name__ == '__main__':
    unittest.main()

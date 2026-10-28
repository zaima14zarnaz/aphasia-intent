# Intent Annotator

A dependency-free website for speech-language pathologists to annotate gesture movement, gesture meaning, and communicative intent in MP4 clips. The desktop workspace puts the video and context on the left and annotation fields on the right; smaller screens stack the panels.

## Start on the server containing your videos

Requires Python 3.10 or newer. From this project folder:

```bash
python3 scripts/prepare_clips.py
python3 -m http.server 8000 --bind 127.0.0.1
```

Open **http://localhost:8000** in a browser on that machine. If this is a remote server, forward port 8000 to your computer with SSH, then open the forwarded address. The preparation script defaults to your directory:

```text
/data/quantization/zaima/aphasiabank/neural2/control/annotations/intent/video_clips
```

It discovers MP4s recursively, copies them into `videos/`, and generates `data/clips.json`. Videos were not available during creation of this package: the supplied manifest contains three clearly labeled example tabs with no video attached.

Do not open `index.html` by double-clicking it; the JavaScript modules and manifest need an HTTP server.

## Use a chosen set of clips

Put one filename per line in a text file, relative to the source directory. Lines beginning with `#` are comments. Your list order determines tab order:

```text
participant01_clip02.mp4
participant03_clip01.mp4
session2/participant04_clip03.mp4
```

Then run:

```bash
python3 scripts/prepare_clips.py --selection selected_clips.txt
```

To use another directory:

```bash
python3 scripts/prepare_clips.py --source /absolute/path/to/video_clips
```

The script preserves edited task names, transcripts, tab titles, example annotations, and placeholder flags for matching clip IDs when rerun. Copies in `videos/` are updated from the source. The source files are never modified. Renaming a clip changes its ID, so keep filenames stable once annotation starts.

## Use local files without copying videos

Select **Choose video folder** on the website. All MP4s in that folder and its subfolders become video tabs. The files stay on the annotator's device and are not uploaded. Reselect the same source folder after reopening the page; existing responses reconnect by relative filename. Choose the `video_clips` folder itself, rather than its parent, to match IDs made by the preparation script.

To retain your chosen set, titles, and transcripts without copying video bytes, generate the metadata first:

```bash
python3 scripts/prepare_clips.py --manifest-only
```

The folder picker loads every MP4 in the selected folder, so use a folder containing only the intended subset if necessary. It reuses metadata for matching clip IDs. A local folder cannot be accessed remotely by another SLP: each SLP must have a copy of the clips or use reachable hosted video URLs.

## Edit transcripts, tasks, and examples

Edit **`data/clips.json`**. Each clip has this structure:

```json
{
  "id": "participant01_clip02.mp4",
  "title": "Clip 01",
  "filename": "participant01_clip02.mp4",
  "src": "videos/participant01_clip02.mp4",
  "task": "Describe a recent activity",
  "transcript": "The written transcript goes here.",
  "placeholder_context": false
}
```

Paths are relative to the website root; URL-encode special characters in `src` (the script does this automatically). Leave `id` unchanged to preserve saved responses. Set `placeholder_context` to `false` once the task and transcript are real.

The top-level **`examples`** object holds the editable example line for each field: `movement`, `meaning`, `intent`, `reasoning`, and `role`. Examples are displayed as guidance below the fields and never prefilled or exported as real annotations. They are generic illustrations, not observations of the participant clips. Review or remove them before an actual study if examples could influence judgments.

## Annotator workflow

1. Enter an annotator ID (for example, `SLP-01`). Use one annotator per browser profile; changing the ID labels exports and does not create a separate response set.
2. Choose a video tab or use **Previous clip** / **Next clip**. Arrow keys, Home, and End also navigate focused tabs.
3. Watch the video and read the transcript and task below it. Playback controls, replay, and speed adjustment are available.
4. Fill in **Movement description**, **Gesture meaning**, **Intent**, and **Reasoning**.
5. Optionally select one or more roles of gesture. Selecting **Unclear** clears the other roles.
6. Select **Mark clip complete**. All four text fields must contain a response. The progress bar counts completed clips; editing a completed response reopens that clip for review.
7. Use **Annotation files → Export JSON / backup** and/or **Export CSV**, then return the downloaded file to the study coordinator.

Drafts and completion status save on each edit in browser local storage, scoped to the site's origin. Navigation and reload retain responses. Clearing browser data removes them; private browsing may not retain them. A storage failure is shown visibly and exports remain available. Avoid editing simultaneously in multiple tabs: if another tab saves changes, this tab pauses browser saving and asks you to export and reload.

**There is no backend or automatic submission.** The coordinator receives annotations only when the annotator sends the exported file. JSON supports restoring responses with **Import JSON backup**; import matches stable clip IDs, ignores clips outside the current set, and asks before replacing existing responses. Export the current set before switching folders, since exports include only the currently displayed clips. Responses for other sets remain in browser storage.

JSON includes annotator ID, clip IDs and filenames, video source (omitted for temporary local-file URLs), task, transcript, four text responses, role IDs, completion, and timestamps. CSV includes the same fields with human-readable role labels. Formula-like text in CSV is prefixed with an apostrophe for safe spreadsheet viewing; JSON preserves the exact text.

## Publish on GitHub Pages

The supplied **`.github/workflows/pages.yml`** builds and deploys on pushes to `main`. No Node package installation or paid API is needed.

1. Create a GitHub repository and put this folder's **contents** at the repository root, including `.github/workflows/pages.yml` and `.gitignore`.
2. Prepare the clip manifest for one of the three media options below.
3. In the repository, open **Settings → Pages → Build and deployment → Source** and choose **GitHub Actions**.
4. Push to `main` or manually run **Deploy Intent Annotator to GitHub Pages** from the Actions tab. If your default branch has another name, update `branches: [main]` in the workflow.
5. Open the deployed URL shown in the workflow's deployment environment.

The workflow uses GitHub's documented Pages actions and minimal deployment permissions. See the official [custom workflow documentation](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages) and [publishing source setup](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site).

### Choose how SLPs will access videos

**A. Hosted video URLs.** Host authorized MP4s on a reachable HTTPS service and generate the manifest on your server:

```bash
python3 scripts/prepare_clips.py --base-url https://your-approved-host.example/video_clips
```

Replace the example host with your real media directory. URLs must serve playable MP4 bytes, not a sharing-page HTML document. Authenticated hosts must permit playback from the website in the SLP's browser. The app adds no authentication of its own. Do not put credentials or expiring access tokens in a public manifest.

**B. Each SLP chooses their local clip folder.** Keep the example manifest for a generic workspace, or run `--manifest-only` for real metadata, then set each clip's `src` to `""` before pushing. Each SLP chooses the distributed folder locally. Keeping empty URLs avoids build failures for missing video files.

**C. Bundle authorized videos with the website.** The default preparation command copies videos into `videos/`. That directory is intentionally ignored by Git. If you have confirmed those videos may be published at the deployed site's access level, add only the intended files explicitly with `git add -f videos/<filename>.mp4`. The Pages build needs the referenced videos to exist in the checkout. Large collections are better served by a separate approved media host.

**A GitHub Pages website cannot read `/data/...` on your server or enumerate its files.** You must prepare a manifest and supply reachable video URLs, bundle the videos, or let the annotator select local files. A private source repository alone does not establish private video access; verify the actual deployed site's audience before exposing participant recordings or transcripts. For restricted study material, use an approved authenticated host or keep clips local.

## Files

| File | Purpose |
| --- | --- |
| `index.html` | Two-column annotation workspace and accessible form |
| `assets/styles.css` | Responsive visual design |
| `assets/app.js` | Video tabs, playback, autosave, completion, import/export |
| `assets/model.mjs` | Annotation validation and export helpers |
| `data/clips.json` | Editable clip list, context, and example lines |
| `scripts/prepare_clips.py` | Discover/select MP4s and prepare the manifest |
| `scripts/build_site.py` | Stage web assets and only referenced local videos in `dist/` |
| `.github/workflows/pages.yml` | GitHub Pages build and deployment |
| `tests/` | Focused annotation/export and preparation checks |

## Verification and troubleshooting

```bash
node --test tests/model.test.mjs
python3 -m unittest discover -s tests -p 'test_*.py'
python3 scripts/build_site.py
```

Node is needed only for the optional JavaScript checks. `dist/` is regenerated for deployment; edit the source files rather than `dist/`.

If a video does not load, check its URL and whether it uses an MP4 codec supported by the annotator's browser (H.264 video with AAC audio is a common choice). The app shows an error without losing text responses. If a Pages build reports a missing local video, commit that authorized video, use an HTTPS source URL, or clear its `src` and use the folder picker. If the clip list fails, check JSON validity and run through an HTTP server. Browsers may require the user to start playback; videos do not autoplay.

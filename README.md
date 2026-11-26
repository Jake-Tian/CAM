# CAM on HippoVlog

CAM builds a memory graph from vlog frames and uses it to answer multiple-choice questions. HippoVlog has 25 videos with 40 questions each, covering visual details, sounds, audio-visual events, and video summaries. [CAM project](https://github.com/Jake-Tian/CAM#readme) · [HippoVlog frames and audio](https://huggingface.co/datasets/JakeTian/HippoVlog)

## Quick start

1. **Check out HippoVlog and create the Conda environment**

   ```bash
   git clone --branch HippoVlog https://github.com/Jake-Tian/CAM.git
   cd CAM
   conda create -n cam-hippovlog python=3.11 pip -y
   conda activate cam-hippovlog
   pip install -r requirements.txt
   ```

2. **Set your API key**

   ```bash
   export OPENAI_API_KEY="your-api-key"
   ```

   This branch makes metered OpenAI API calls to `gpt-5-mini` for frames, `gpt-4o-mini` for text generation and QA, and `text-embedding-3-small` for embeddings.

3. **Run HippoVlog**

   With Bash 4 or later, run the full benchmark (25 videos, 1,000 questions) with up to four videos in parallel by default.

   ```bash
   bash run_video_pipeline.sh
   ```

   For a subset, pass video IDs. This example runs two videos in parallel.

   ```bash
   MAX_PARALLEL_JOBS=2 bash run_video_pipeline.sh 1cU-k5d1DuU 6Z_qEtbmK34
   ```

   To run one video and its 40 questions, use `bash run_video_pipeline.sh 1cU-k5d1DuU`. Set `MAX_PARALLEL_JOBS=1` for sequential processing. [Video IDs](video_list.txt)

The runner downloads the shared audio folder once per invocation and the frames for each requested video.

## Inputs and outputs

| Path | Contents |
| --- | --- |
| [`data/questions.jsonl`](data/questions.jsonl) | One question per line with `video_id`, `question_text`, choices A through D in `options`, and the reference letter in `correct_answer` |
| `data/frames/<video_id>/<clip_id>/*.jpg` | Clip frames used to build the graph and revisit visual evidence |
| `data/audio/<video_id>.wav` | Full audio track used by the listening tool |
| `data/graphs/<video_id>.pkl` | Saved memory graph |
| `data/memorization/<video_id>.json` | Per-clip observations and token counts for graph construction |
| `data/reasoning/<video_id>.json` | Per-question predictions, reference letters, `evaluate_correct` verdicts, and reasoning traces |

`reason.py` selects questions by `video_id` and scores each predicted option letter by exact match against `correct_answer`. [Scoring code](reason.py)

For questions about sound, the agent exposes a `listen_to_audio` tool. Using it with MoviePy 2 requires changing the clip-slicing call from `subclip()` to `subclipped()`. The tool also sends WAV data to the default vision model, `gpt-5-mini`. Audio queries require a model and request format that accept WAV input. [Audio tool](utils/search_tools.py) · [Model helper](utils/mllm_gpt.py) · [MoviePy clip API](https://zulko.github.io/moviepy/reference/reference/moviepy.Clip.Clip.html#moviepy.Clip.Clip.subclipped) · [Model input types](https://developers.openai.com/api/docs/models/gpt-5-mini)

## Repeat QA with a saved graph

The runner removes each video's frame directory after processing and keeps the audio, graph, and JSON outputs. To repeat QA with frame rewatching, restore the frames and run the question-answering step directly. This rewrites the per-video reasoning JSON.

```bash
python preprocessing/download_hf_frames.py 1cU-k5d1DuU
python reason.py 1cU-k5d1DuU
```

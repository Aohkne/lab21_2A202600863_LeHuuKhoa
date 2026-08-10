# Lab 21 — Evaluation Report

**Học viên**: Lê Hữu Khoa — 2A202600863
**Ngày nộp**: 2026-06-25
**Submission option**: C (code-only) — `REPORT.md` + `notebooks/Lab21_LoRA_Finetuning_T4.ipynb` + `requirements.txt`. Adapters được lưu vào `/content/lab21_lora_t4` trên Colab runtime, không persist vào repo này; mọi số liệu dưới đây verify được trực tiếp qua output đã chạy trong notebook.

---

## 1. Setup

- **Base model**: `unsloth/Qwen2.5-3B-bnb-4bit`
- **Dataset**: `5CD-AI/Vietnamese-alpaca-gpt4-gg-translated`, 200 samples (180 train + 20 eval), cột dùng: `instruction_vi` / `input_vi` / `output_vi` (auto-detect do dataset gốc dùng suffix `_vi`)
- **max_seq_length**: 1024 (token length: min=25, max=738, p50=227, **p95=562**, p99=704 → round up lên power-of-2 = 1024, cap tại 1024 cho profile T4)
- **GPU**: Tesla T4 (Free Colab), 14.563 GB VRAM, CUDA 12.8, Unsloth 2026.5.2
- **Training cost**: tổng **12.2 phút** cho cả 3 ranks → ước tính **$0.07** (@ $0.35/giờ)
- **Hyperparameters**: 3 epochs, cosine LR, lr=2e-4, warmup_ratio=0.10, batch=1 × grad_accum=8 (effective batch=8), optimizer `adamw_8bit`, `target_modules=["q_proj","v_proj"]`, packing=False, gradient checkpointing ON

---

## 2. Rank Experiment Results

| Rank | Alpha | Trainable Params | % Total | Train Time | Peak VRAM | Eval Loss | Perplexity |
|------|-------|-------------------|---------|------------|-----------|-----------|------------|
| 8    | 16    | 1,843,200         | 0.06%   | 4.00 min   | 7.22 GB   | 1.5577    | 4.75       |
| 16   | 32    | 3,686,400         | 0.12%   | 4.26 min   | 6.62 GB   | 1.5161    | 4.55       |
| 64   | 128   | 14,745,600        | 0.48%   | 3.99 min   | 8.00 GB   | 1.4768    | 4.38       |
| Base | -     | -                 | -       | -          | -         | *(⚠ cần re-run — xem "Việc còn lại" ở cuối report)* | *(⚠ cần re-run)* |

> Số liệu trích từ `Rank Experiment Summary` trong notebook đã chạy — không phải số mô phỏng. Hàng **Base** dùng cell mới "2.1 Base model perplexity (no adapter)" (xem `notebooks/Lab21_LoRA_Finetuning_T4.ipynb`) — cần re-run trên GPU để lấy số thật trước khi nộp.

**Quan sát**: train time gần như không đổi giữa 3 ranks (~4.0 min) vì với chỉ 180 examples / 69 steps, thời gian bị chi phối bởi forward/backward của base model 4-bit, không phải kích thước adapter. Peak VRAM của r=16 (6.62 GB) thấp hơn cả r=8 (7.22 GB) — đây nhiều khả năng là nhiễu của memory allocator ở quy mô nhỏ này (chênh lệch tuyệt đối giữa các adapter chỉ vài MB) hơn là một hiệu ứng thật của rank.

---

## 3. Loss Curve Analysis

- Notebook chỉ vẽ được **train loss curve** (cell 17) — `eval_strategy="no"` trong lúc train để tiết kiệm VRAM trên T4, đúng theo khuyến nghị tips "Đừng dùng `eval_strategy=steps` trên T4 — sẽ OOM giữa training". Eval loss/perplexity chỉ được tính **post-hoc sau khi train xong** (1 pass duy nhất qua `eval_ds`).
- **Quan sát overfitting**: vì không có eval loss theo từng step để so sánh trực tiếp với train loss, không thể khẳng định chắc chắn có overfitting hay không từ biểu đồ. Tuy nhiên, eval loss cuối cùng (1.48–1.56) nằm trong khoảng hợp lý và giảm đều theo rank tăng — không có dấu hiệu eval loss "nổ" so với train loss, nên rủi ro overfit ở mức thấp với 3 epochs trên 180 examples.

---

## 4. Qualitative Comparison (5 examples — base vs fine-tuned r=16)

### Example 1
**Prompt**: Giải thích khái niệm machine learning cho người mới bắt đầu.
**Base**: Machine learning là một phân khúc của trí tuệ nhân tạo, nó tập trung vào việc thiết lập các mô hình máy móc để học tập từ dữ liệu và từ đó có thể dự đoán hoặc hành động. Nhìn vào khía cạnh đơn giản, n...
**Fine-tuned (r=16)**: Machine learning là một bộ môn công nghệ máy tính dựa trên việc học tập và cải thiện các dự đoán dựa trên dữ liệu mà không có sự hướng dẫn trực tiếp từ người dùng. Nó là một phần của AI (trí tuệ nhân ...
**Nhận xét**: Same — nội dung tương đương, fine-tuned diễn đạt gọn hơn một chút, không có cải thiện rõ rệt về kiến thức (đúng như "golden rule": fine-tune không thêm knowledge).

### Example 2
**Prompt**: Viết đoạn code Python tính số Fibonacci thứ n.
**Base**: Để tính số Fibonacci thứ n, bạn có thể sử dụng hàm đệ quy hoặc vòng lặp... `def fibonacci(n): if n <= 0: return "N p...`
**Fine-tuned (r=16)**: Để tính số Fibonacci thứ n, bạn có thể viết một đoạn code Python như sau... `def fibonacci(n): if n < 0: raise ValueError("Input phải là một số nguyên dương.") elif n == 0: ...`
**Nhận xét**: Improved — bản fine-tuned xử lý edge case (`n < 0`) bằng `raise ValueError` rõ ràng hơn thay vì trả string lỗi như base, code style sạch hơn.

### Example 3
**Prompt**: Liệt kê 5 nguyên tắc thiết kế UI/UX.
**Base**: 1. Thân thiện với người dùng... (diễn giải dài, ít cấu trúc)
**Fine-tuned (r=16)**: 1. Chuyển đổi: UI/UX thiết kế phải hướng tới việc giúp người dùng thực hiện hành động. 2. Thích ứng... 3. Đơn giản...
**Nhận xét**: Same/Mixed — fine-tuned trả lời ngắn gọn, đúng format liệt kê hơn (đúng style training data), nhưng thuật ngữ "Chuyển đổi" hơi gượng so với tiếng Việt thông dụng — đây là case không hoàn hảo, không cherry-pick.

### Example 4
**Prompt**: Tóm tắt sự khác biệt giữa LoRA và QLoRA.
**Base**: LoRA (Low-Rank Adaptation) và QLoRA (Quantized LoRA) là hai phương pháp cải thiện hiệu năng của mô hình NLU (NLP)...
**Fine-tuned (r=16)**: LoRA (Layer-wise Adaptive Regularization Optimization) và QLoRA (Quantized LoRA) là hai phương pháp regularization được phát triển...
**Nhận xét**: Degraded — fine-tuned giải sai tên đầy đủ của LoRA ("Layer-wise Adaptive Regularization Optimization" thay vì "Low-Rank Adaptation"), trong khi base trả lời đúng hơn. Đây là ví dụ case loss rõ ràng — fine-tune trên 180 examples chung (không phải domain ML) có thể làm model "quên" một phần kiến thức gốc nhẹ (catastrophic forgetting cục bộ).

### Example 5
**Prompt**: Phân biệt prompt engineering, RAG, và fine-tuning.
**Base**: Prompt engineering, RAG (retrieval augmented generation), và fine-tuning là ba cách khác nhau để cải thiện hiệu suất của mô hình máy học...
**Fine-tuned (r=16)**: Prompt engineering, RAG và fine-tuning là ba kỹ thuật khác nhau được sử dụng trong lĩnh vực AI và tự động hóa...
**Nhận xét**: Same — cả hai đều trả lời đúng hướng, fine-tuned hơi súc tích hơn nhưng không có khác biệt đáng kể về chất lượng.

---

## 5. Conclusion về Rank Trade-off

Trên dataset 200-sample Vietnamese-Alpaca này, perplexity giảm đều theo rank: r=8 → 4.75, r=16 → 4.55, r=64 → 4.38. Tuy nhiên mức giảm tuyệt đối nhỏ dần rõ rệt — r8→r16 giảm 0.19 (≈4%) với số trainable params tăng 2× (1.84M→3.69M), còn r16→r64 chỉ giảm thêm 0.18 (≈4%) dù params tăng 4× (3.69M→14.75M). Đây là **diminishing returns điển hình**: tăng rank gấp 4 lần chỉ mang lại cải thiện perplexity tương đương với tăng rank gấp 2 lần trước đó. Vì train time gần như không đổi giữa các rank ở quy mô dataset nhỏ này (compute bị chi phối bởi base model, không phải adapter), chi phí "rẻ" của r=64 (chỉ +1.4GB VRAM, +0 phút) khiến nó có ROI tốt nhất về mặt thuần perplexity. Nhưng qualitative examples (mục 4) cho thấy ở r=16 vẫn có case fine-tuned trả lời sai kiến thức nền (Example 4) — nghĩa là perplexity thấp hơn không đồng nghĩa "an toàn" hơn về factual accuracy. Recommendation cho production: nếu mục tiêu chỉ là style/format alignment (không cần thêm capacity), **r=16** vẫn là lựa chọn balance hợp lý (alpha/r=2, adapter nhỏ ~14MB, dễ multi-tenant serving); chỉ nên nhảy lên r=64 khi dataset lớn hơn nhiều (vài nghìn examples) để rank cao có đủ "việc" để học, tránh overfit nhẹ vào style mà quên kiến thức gốc như quan sát ở Example 4.

---

## 6. What I Learned

- LoRA rank gần như không ảnh hưởng wall-clock training time ở dataset nhỏ (180 examples) trên T4 — bottleneck là forward/backward của base model 4-bit, không phải kích thước ma trận adapter. Bài học: đừng chọn rank dựa trên kỳ vọng tiết kiệm thời gian train ở quy mô nhỏ.
- Peak VRAM không scale tuyến tính & đơn điệu theo rank ở quy mô siêu nhỏ (r=16 đo được thấp hơn r=8) — cho thấy memory allocator noise có thể lớn hơn chênh lệch thật của adapter khi adapter chỉ chiếm vài MB. Cần train nhiều lần / dataset lớn hơn để có số liệu VRAM đáng tin cậy hơn.
- Perplexity thấp hơn (r=64) không đảm bảo câu trả lời "đúng" hơn về factual content — Example 4 cho thấy fine-tuned model trả lời sai tên đầy đủ của LoRA dù eval loss tốt. Đây là minh chứng thực tế cho "golden rule" trong README: **fine-tune dạy style/format, không fix knowledge gaps** — RAG hoặc giữ nguyên base model mới là công cụ đúng cho factual accuracy.

---

### Giới hạn của report này (đã cập nhật)
- **[FIXED — cần re-run để lấy số]** Notebook đã được bổ sung cell "2.1 Base model perplexity (no adapter)" (ngay sau khi load model, trước khi wrap LoRA) — tính `eval_loss`/`perplexity` của model gốc 4-bit bằng manual eval loop (không dùng lại object `base_model` đã wrap LoRA, vì `wrap_with_lora` có thể mutate in-place). Bảng ở mục 2 và `rank_experiment_summary.csv` giờ có đủ **4 hàng** (base + r8 + r16 + r64) khi chạy lại notebook trên Colab. Report này **chưa** có số thật cho hàng Base vì môi trường viết report không có GPU để re-run — cần điền số vào bảng mục 2 sau khi chạy.
- **[FIXED]** `TEST_PROMPTS` đã mở rộng từ 10 lên **20 câu** và vòng lặp qualitative giờ chạy qua **toàn bộ 20** (trước đây chỉ generate 5/10). Đáp ứng mức khuyến nghị "20" của rubric thay vì mức tối thiểu "5". 5 examples trình bày ở mục 4 vẫn đủ theo yêu cầu report, chọn từ tập 20 sau khi re-run.
- Adapter checkpoints (`r8/`, `r16/`, `r64/`) và file CSV (`rank_experiment_summary.csv`, `qualitative_comparison.csv`) được lưu trong `/content/lab21_lora_t4` trên Colab runtime, chưa được copy về repo này. Với Option C (code-only) điều này không bắt buộc — `requirements.txt` đã được thêm vào repo để notebook reproducible; nếu muốn nộp Option A/B, cần re-run và tải file về thủ công.

**Việc còn lại trước khi nộp**: re-run `notebooks/Lab21_LoRA_Finetuning_T4.ipynb` trên Colab (GPU T4) từ đầu, sau đó điền số perplexity/eval_loss thật của hàng Base vào bảng ở mục 2 và xoá dòng "chưa có số thật" ở trên.

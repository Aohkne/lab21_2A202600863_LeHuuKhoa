# Lab 21 — Evaluation Report

**Học viên**: Lê Hữu Khoa — 2A202600863
**Ngày nộp**: 2026-08-10
**Submission option**: C (code-only) — `REPORT.md` + `notebooks/Lab21_LoRA_Finetuning_T4.ipynb` (đã chạy end-to-end, giữ nguyên outputs làm bằng chứng) + `requirements.txt`. Adapters được lưu vào `/content/lab21_lora_t4` trên Colab runtime, không persist vào repo này; mọi số liệu dưới đây trích trực tiếp từ output đã chạy trong notebook — không có số mô phỏng.

---

## 1. Setup

- **Base model**: `unsloth/Qwen2.5-3B-bnb-4bit`
- **Dataset**: `5CD-AI/Vietnamese-alpaca-gpt4-gg-translated`, 200 samples (180 train + 20 eval), cột dùng: `instruction_vi` / `input_vi` / `output_vi` (auto-detect do dataset gốc dùng suffix `_vi`)
- **max_seq_length**: 1024 (token length: min=25, max=738, p50=227, **p95=562**, p99=704 → round up lên power-of-2 = 1024, cap tại 1024 cho profile T4)
- **GPU**: Tesla T4 (Free Colab), 15.6 GB VRAM (Unsloth báo max usable ≈14.6 GB), CUDA 12.8, PyTorch 2.11.0+cu128, Unsloth 2026.8.10, Transformers 5.5.0
- **Training cost**: tổng **17.0 phút** cho cả 3 ranks (base model chỉ eval, không train) → ước tính **$0.10** (@ $0.35/giờ)
- **Hyperparameters**: 3 epochs, cosine LR, lr=2e-4, warmup_ratio=0.10, batch=1 × grad_accum=8 (effective batch=8), optimizer `adamw_8bit`, `target_modules=["q_proj","v_proj"]`, packing=False, gradient checkpointing ON

---

## 2. Rank Experiment Results

| Rank | Alpha | Trainable Params | % Total | Train Time | Peak VRAM | Eval Loss | Perplexity |
|------|-------|-------------------|---------|------------|-----------|-----------|------------|
| Base | -     | 0                 | -       | – (không train) | – (chỉ eval, không train nên không đo peak train VRAM) | 1.8840 | 6.58 |
| 8    | 16    | 1,843,200         | 0.06%   | 5.80 min   | 3.22 GB   | 1.5575    | 4.75       |
| 16   | 32    | 3,686,400         | 0.12%   | 5.62 min   | 2.62 GB   | 1.5160    | 4.55       |
| 64   | 128   | 14,745,600        | 0.48%   | 5.56 min   | 4.00 GB   | 1.4769    | 4.38       |

> Số liệu trích từ `Rank Experiment Summary (incl. base)` — cell "Build summary table" — trong notebook đã chạy đầy đủ trên Tesla T4 (không có lỗi, không có OOM fallback). Hàng **Base** dùng cell "2.1 Base model perplexity (no adapter)": load lại một instance model 4-bit riêng biệt (không tái dùng object đã bị `wrap_with_lora` gắn adapter), chạy manual eval loop batch=1 qua `eval_ds`, tính `exp(eval_loss)`.

**Quan sát**:
- **Fine-tuning (ở bất kỳ rank nào) đóng góp phần lớn giá trị, không phải việc tăng rank.** Perplexity giảm từ 6.58 (base) xuống 4.75 chỉ với r=8 — đã đạt **83%** tổng mức cải thiện có thể có trong thí nghiệm này (base → r=64: 6.58 → 4.38, giảm 2.20; riêng base → r=8 đã giảm 1.83). Từ r=8 lên r=64 (tăng gấp **8×** trainable params: 1.84M → 14.75M) chỉ "vét" thêm 17% phần còn lại của cải thiện.
- **Train time gần như không đổi giữa 3 ranks** (~5.6–5.8 phút) vì với chỉ 180 examples / 69 steps, thời gian bị chi phối bởi forward/backward của base model 4-bit, không phải kích thước adapter.
- **Peak VRAM của r=16 (2.62 GB) thấp hơn r=8 (3.22 GB) — và hiện tượng này lặp lại giống hệt ở một lần chạy độc lập trước đó** (chênh lệch ~0.60 GB cả hai lần, dù giá trị tuyệt đối khác nhau do thư viện Unsloth được nâng cấp giữa hai lần chạy — xem mục 6). Vì tái hiện được ở 2 session độc lập với cùng một chiều lệch, đây nhiều khả năng **không phải nhiễu ngẫu nhiên** mà là chi phí hệ thống của lần `train_one_rank()` đầu tiên sau khi dọn baseline (model reload + Triton kernel autotune lần đầu), chứ không phải hiệu ứng thật của kích thước adapter.

---

## 3. Loss Curve Analysis

- Notebook chỉ vẽ được **train loss curve** (mục 3, sau khi train baseline r=16) — `eval_strategy="no"` trong lúc train để tiết kiệm VRAM trên T4, đúng theo khuyến nghị tips "Đừng dùng `eval_strategy=steps` trên T4 — sẽ OOM giữa training". Eval loss/perplexity chỉ được tính **post-hoc sau khi train xong** (1 pass duy nhất qua `eval_ds`).
- **Quan sát overfitting**: vì không có eval loss theo từng step để so sánh trực tiếp với train loss, không thể khẳng định chắc chắn có overfitting hay không từ biểu đồ train-loss-only. Tuy nhiên, eval loss cuối cùng cho cả 3 ranks (1.48–1.56) đều **thấp hơn nhiều** so với eval loss của base model (1.884) — hướng đi đúng, không có dấu hiệu eval loss "nổ ngược" so với train loss. Với chỉ 3 epochs trên 180 examples và LoRA rank thấp (0.06–0.48% tham số), rủi ro overfit nặng ở mức thấp; rủi ro thực tế quan sát được lại nằm ở **catastrophic forgetting cục bộ** (xem Example 4, mục 4) chứ không phải overfit theo nghĩa cổ điển.

---

## 4. Qualitative Comparison (5 examples — base vs fine-tuned r=16)

> Trích trực tiếp từ output notebook (`generate_response`, temperature=0.7, top_p=0.9, do_sample=True — text đầy đủ, không rút gọn thêm ngoài truncation 200 ký tự có sẵn trong log notebook), chọn từ tập 20 test prompts, đảm bảo có cả case win lẫn case loss.

### Example 1 — Win rõ rệt
**Prompt**: Viết đoạn code Python tính số Fibonacci thứ n.
**Base**: "Để tính số Fibonacci thứ n, bạn có thể sử dụng hàm đệ quy hoặc vòng lặp. Đây là một đoạn mã Python cho phép bạn tính số Fibonacci thứ n: ..."
**Fine-tuned (r=16)**: "Để tính số Fibonacci thứ n, bạn có thể viết một đoạn code Python như sau: `def fibonacci(n): if n < 0: raise ValueError("Input phải là một số nguyên dương.") elif n == 0: ...`"
**Nhận xét**: Improved — bản fine-tuned xử lý edge case (`n < 0`) bằng `raise ValueError` rõ ràng hơn, code style sát với format trong training data hơn.

### Example 2 — Win rõ rệt (base hallucinate hoàn toàn)
**Prompt**: Khi nào nên tăng LoRA rank từ 16 lên 64?
**Base**: "LoRA rank là một hệ thống điểm đánh giá cho các nhân viên của một công ty, có thể được tăng hoặc giảm dựa trên các hành động và sự đóng góp của họ. Tăng LoRA rank từ 16 lên 64 thường chỉ xảy ra khi nh..."
**Fine-tuned (r=16)**: "LoRA (Low-Rank Approximation) là một kỹ thuật được sử dụng để giảm kích thước mô hình mạng neural network (NN) bằng cách thay thế một số trọng số với một ma trận nhỏ hơn. Tuy nhiên, ..."
**Nhận xét**: Improved rõ rệt — base hoàn toàn hiểu sai thuật ngữ "LoRA rank" (nhầm thành hệ thống đánh giá nhân viên!), trong khi fine-tuned bắt đúng ngữ cảnh kỹ thuật (dù gọi sai "Approximation" thay vì "Adaptation" — vẫn còn lỗi nhỏ, nhưng đúng hướng hoàn toàn so với base).

### Example 3 — Loss rõ ràng (không cherry-pick)
**Prompt**: Tóm tắt sự khác biệt giữa LoRA và QLoRA.
**Base**: "LoRA (Low-Rank Adaptation) và QLoRA (Quantized LoRA) là hai phương pháp cải thiện hiệu suất của mô hình NLP được xây dựng trên nền tảng Transformers. LoRA là phương pháp cải thiện hiệu suất của mô hìn..."
**Fine-tuned (r=16)**: "LoRA (Large Language Model Adapter) và QLoRA (Quantized LoRA) là hai phương pháp tối ưu hóa mô hình ngôn ngữ lớn (LM) được phát triển để cải thiện hiệu suất của các LM lớn trong việc giải quyết các vấ..."
**Nhận xét**: Degraded — base gọi đúng tên đầy đủ "Low-Rank Adaptation", fine-tuned lại đổi thành "Large Language Model Adapter" (sai). Đây là case loss rõ ràng — fine-tune trên 180 examples đa chủ đề (không phải domain ML) làm model "quên" một phần kiến thức gốc (catastrophic forgetting cục bộ), dù eval loss/perplexity tổng thể tốt hơn base.

### Example 4 — Mixed/loss (cả hai đều sai kiến thức nền)
**Prompt**: So sánh ưu nhược điểm giữa DPO và SFT.
**Base**: "Điểm mạnh của DPO (Đảo ngược Phương trình Động): DPO được đánh giá cao về khả năng hiệu chỉnh và điều chỉnh nhanh chóng. DPO có thể phản ứng với các thay đổi môi trường nhanh hơn và chính xác hơn vì n..."
**Fine-tuned (r=16)**: "DPO (Dynamic Period of Inactivity) và SFT (Single File Transfer) là hai phương pháp khác nhau để phân chia dữ liệu trên một mạng. DPO là một phương pháp phân chia dữ liệu dựa trên thời gian, trong khi..."
**Nhận xét**: Cả hai đều sai hoàn toàn thuật ngữ (đúng ra DPO = Direct Preference Optimization, SFT = Supervised Fine-Tuning) — dataset training không chứa domain LLM/ML nên cả base lẫn fine-tuned đều hallucinate. Minh chứng rằng fine-tune trên dataset generic không tạo ra kiến thức mới, kể cả khi loss giảm.

### Example 5 — Same/Mixed
**Prompt**: Liệt kê 5 nguyên tắc thiết kế UI/UX.
**Base**: "1. Thân thiện với người dùng: Mục đích của thiết kế UI/UX là cung cấp trải nghiệm người dùng thân thiện, dễ sử dụng và thân thiện. Điều này có thể được thể hiện thông qua việc sắp xếp bố cục, màu sắc,..."
**Fine-tuned (r=16)**: "1. Chuyển đổi: UI/UX thiết kế phải hướng tới việc giúp người dùng thực hiện hành động. 2. Thích ứng: UI/UX thiết kế phải thích ứng với nhiều loại thiết bị và kích thước màn hình. 3. Đơn giản: UI/UX th..."
**Nhận xét**: Mixed — fine-tuned trả lời đúng format liệt kê ngắn gọn hơn (đúng style training data), nhưng thuật ngữ "Chuyển đổi" hơi gượng so với tiếng Việt thông dụng cho khái niệm UX — không hoàn hảo, giữ lại để tránh cherry-pick.

---

## 5. Conclusion về Rank Trade-off

Trên dataset 200-sample Vietnamese-Alpaca này, cả 4 điểm dữ liệu (base + r=8 + r=16 + r=64) cho thấy một pattern rõ ràng: **việc fine-tune, ở bất kỳ rank nào, mới là yếu tố tạo ra phần lớn giá trị — không phải việc tăng rank**. Perplexity giảm mạnh từ 6.58 (base, chưa fine-tune) xuống 4.75 chỉ với r=8 (giảm 1.83, tương đương 83% tổng mức cải thiện đo được trong thí nghiệm), trong khi tăng rank thêm 8× (r=8 → r=64, 1.84M → 14.75M tham số) chỉ mang lại thêm 0.37 điểm perplexity (17% phần còn lại). Đây là **diminishing returns rất rõ nét**: chi phí biên (thêm tham số, VRAM) tăng gần tuyến tính theo rank, nhưng lợi ích biên giảm mạnh sau bước nhảy đầu tiên từ "không fine-tune" sang "có fine-tune". Vì train time gần như không đổi giữa các rank ở quy mô dataset nhỏ này (compute bị chi phối bởi base model 4-bit, không phải adapter) và VRAM chênh lệch giữa các rank chỉ ~1.4 GB, chi phí tuyệt đối của r=64 vẫn "rẻ" — nhưng qualitative examples (mục 4) cho thấy perplexity thấp hơn không đồng nghĩa "đúng" hơn về factual accuracy: cả base lẫn fine-tuned r=16 đều mắc lỗi kiến thức nền ở các câu hỏi ngoài phạm vi dataset (Example 3, 4), và một số case fine-tuned còn sai hơn base (catastrophic forgetting cục bộ). Recommendation cho production: với dataset nhỏ (180 examples) và mục tiêu chính là style/format alignment, **r=8** đã đạt ROI tốt nhất (83% cải thiện với 12.5% số tham số so với r=64, adapter nhỏ nhất, dễ multi-tenant serving nhất); chỉ nên nhảy lên r=16 hoặc r=64 khi dataset lớn hơn nhiều (vài nghìn examples, đa dạng hơn) để rank cao có đủ "việc" để học mà không đánh đổi kiến thức nền, và luôn kết hợp với RAG/giữ nguyên base model cho các câu hỏi cần factual accuracy cao.

---

## 6. What I Learned

- **Fine-tune (ở rank thấp nhất) mang lại phần lớn lợi ích; tăng rank tiếp theo có lợi ích biên giảm rất nhanh.** Trước khi có số base model, tôi từng nghĩ r=64 "rẻ nên cứ chọn cao nhất" — nhưng khi có đủ 4 điểm dữ liệu (base, r8, r16, r64), rõ ràng 83% cải thiện đã nằm trong bước "có fine-tune hay không", không phải "rank bao nhiêu". Bài học: luôn đo baseline (chưa fine-tune) trước khi so sánh các rank với nhau, nếu không dễ đánh giá sai ROI của việc tăng rank.
- **Peak VRAM "bất thường" (r=8 > r=16) tái hiện giống hệt ở 2 lần chạy độc lập (chênh lệch ~0.60 GB cả hai lần)** — ban đầu tôi nghĩ đây là nhiễu ngẫu nhiên của memory allocator, nhưng vì nó lặp lại đúng chiều ở phiên chạy sau (dù giá trị tuyệt đối khác do Unsloth được nâng cấp từ 2026.5.2 lên 2026.8.10 giữa 2 lần chạy, làm VRAM giảm gần 2× nhờ tối ưu mới), khả năng cao đây là chi phí hệ thống cố định của lần `train_one_rank()` đầu tiên trong loop (model reload + kernel autotune lần đầu), không phải hiệu ứng thật của rank. Bài học: một quan sát "noise" chỉ nên kết luận là noise sau khi thử tái hiện — nếu nó lặp lại có hệ thống, cần tìm nguyên nhân thay vì gạt bỏ.
- Perplexity thấp hơn không đảm bảo câu trả lời "đúng" hơn về factual content — cả base lẫn fine-tuned đều hallucinate với các câu hỏi ngoài phạm vi dataset training (ví dụ DPO/SFT ở Example 4), và fine-tuned đôi khi còn sai hơn base ở kiến thức đã có sẵn đúng (Example 3, tên đầy đủ LoRA). Đây là minh chứng thực tế cho "golden rule" trong README: **fine-tune dạy style/format, không fix knowledge gaps** — RAG hoặc giữ nguyên base model mới là công cụ đúng cho factual accuracy.

---

### Ghi chú xác thực

Notebook `notebooks/Lab21_LoRA_Finetuning_T4.ipynb` đã chạy **end-to-end trên Tesla T4 (Colab)**, tất cả 33 cells thực thi tuần tự không lỗi, không có OOM fallback (`safe_evaluate()` không cần rơi vào nhánh manual-eval dự phòng). Bảng 4-số (base + 3 ranks) và 20/20 qualitative prompts đều có output thật, verify được trực tiếp trong notebook đã lưu (outputs được giữ nguyên, không strip, để làm bằng chứng — phù hợp với Option C khi không có CSV/adapter đính kèm).

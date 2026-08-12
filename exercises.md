# Day 14 — Exercises

## AI Evaluation & Benchmarking · Lab Worksheet

**Thời gian làm bài:** 14:15–17:00

**Domain:** OrbitTech Store Customer Support

Điền trực tiếp câu trả lời vào file này. Golden dataset 20 QA được viết một lần
duy nhất trong `golden_dataset.json`, không chép lại toàn bộ vào Markdown.

---

Từ 14:15–14:30, cài môi trường và chạy baseline tests theo `guide_lab.md`.

---

## Part 1 — Warm-up (14:30–14:45)

### Exercise 1.1 — RAGAS Metric Thresholds

Theo bài giảng:

- 0.8–1.0: Good — monitor, maintain.
- 0.6–0.8: Needs work — analyze failures, iterate.
- Dưới 0.6: Significant issues — investigate.

Với từng metric, xác định khi nào score thấp có thể chấp nhận và khi nào là
critical.

| Metric | Acceptable Low Score Scenario | Critical Low Score Scenario | Action Required |
|---|---|---|---|
| Faithfulness | Khi câu trả lời dùng từ ngữ lịch sự bổ trợ ("Chào bạn!"), từ ngữ nối hoặc kiến thức bổ trợ hợp lý không làm thay đổi bản chất thông tin trong context | Khi model tự bịa ra thông tin sai thực tế (hallucination) về chính sách, giá cả, thời gian bảo hành không có trong context | Siết chặt system prompt về grounding ("Chỉ trả lời dựa trên context được cung cấp"); chèn hallucination guardrail detector. |
| Answer Relevance | Khi câu hỏi phức tạp/ambiguous cần câu trả lời kèm thông tin hướng dẫn mở đầu/từ chối lịch sự làm giảm mật độ từ khóa trùng với câu hỏi | Trả lời lạc đề hoàn toàn (off-topic), lan man không đúng trọng tâm thắc mắc của khách hàng (ví dụ hỏi bảo hành lại trả lời cấu hình) | Tinh chỉnh prompt hướng dẫn trả lời trực diện; cải thiện Intent Classifier / Query Rewriter. |
| Context Recall | Khi expected answer chứa thông tin nâng cao/ngoại lệ mà corpus hiện tại chưa cập nhật, hoặc câu hỏi out-of-scope/adversarial | Retriever bỏ sót các chunk tài liệu chứa thông tin cốt lõi (evidence) cần thiết để trả lời câu hỏi 2-3 tài liệu | Tăng top_k retrieval, cải thiện chunking strategy, dùng hybrid search (BM25 + Vector) hoặc bổ sung tài liệu vào KB. |
| Context Precision | Các chunks được retrieve đều chứa thông tin hữu ích nhưng chunk chứa thông tin quan trọng nhất đứng ở vị trí 2-3 thay vì vị trí 1 | Top 1-2 chunks trả về chứa toàn thông tin nhiễu/không liên quan, đẩy thông tin quan trọng xuống vị trí quá sâu (Lost in the middle) | Sử dụng Cross-Encoder Reranker để sắp xếp lại vị trí chunk; điều chỉnh similarity threshold của Vector DB. |
| Completeness | Khi người dùng yêu cầu câu trả lời tóm tắt ngắn gọn (ví dụ: "chỉ trả lời Yes/No trong 1 câu"), bỏ qua các chi tiết phụ không bắt buộc | Bỏ sót các điều kiện bắt buộc, ngày hiệu lực hoặc ngoại lệ chính sách quan trọng (ví dụ: thiếu điều kiện tem niêm phong khi đổi trả) | Cải thiện Prompt Generation yêu cầu duyệt qua toàn bộ các điều kiện; kiểm tra và nâng cao Context Recall của Retriever. |

### Exercise 1.2 — Bias trong LLM-as-a-Judge

Ba bias thường gặp:

- Position bias: judge ưu tiên answer xuất hiện trước.
- Verbosity bias: judge ưu tiên answer dài hơn.
- Self-preference: judge ưu tiên output giống chính model đó.

**Câu 1: Thiết kế experiment phát hiện position bias với ít nhất hai conditions.**

> *Câu trả lời:*
> Để phát hiện Position Bias của LLM-as-a-Judge, ta thực hiện thí nghiệm A/B testing hoán đổi vị trí (Position Swapping Experiment) với 2 conditions như sau:
> 
> 1. **Condition 1 (Original Order A vs B):** Cho LLM Judge đánh giá hai câu trả lời cho cùng 1 câu hỏi theo thứ tự: `[System Prompt + Rubric + Question] -> Option 1: Answer A, Option 2: Answer B`. Ghi nhận kết quả lựa chọn câu trả lời tốt hơn (Win/Loss).
> 2. **Condition 2 (Swapped Order B vs A):** Giữ nguyên toàn bộ Prompt, Rubric và Question, chỉ hoán đổi vị trí hai câu trả lời: `Option 1: Answer B, Option 2: Answer A`. Ghi nhận kết quả lựa chọn.
> 
> **Đánh giá & Định lượng:** 
> - Đo chỉ số **Position Consistency Rate (PCR)**: $PCR = \frac{\text{Số lần kết luận không đổi (Answer nào thắng vẫn thắng bất kể vị trí)}}{\text{Tổng số cặp test}}$.
> - Nếu $PCR < 0.85$ hoặc tỷ lệ lựa chọn Option 1 luôn vượt trội ($> 65\%$) dù nội dung hoán đổi, kết luận LLM Judge bị ảnh hưởng bởi Position Bias nặng.
> - **Biện pháp khắc phục:** Chạy đồng thời cả 2 chiều (A/B và B/A) và lấy kết quả đồng thuận (Consensus/Majority Vote) hoặc trung bình điểm của cả 2 lần chạy.

**Câu 2: Làm thế nào giảm verbosity bias bằng rubric design?**

> *Câu trả lời:*
> Giảm Verbosity Bias bằng cách thiết kế Rubric tập trung vào **mật độ thông tin (information density)** và **tính chính xác/ngắn gọn (conciseness)** thay vì độ dài văn bản:
> 
> 1. **Quy định tiêu chí phán quyết dựa trên Claim Matching:** Hướng dẫn Judge thực hiện bước trung gian: *"Hãy trích xuất danh sách các ý chính (claims/facts) có trong câu trả lời. Chỉ cộng điểm cho các ý đúng và liên quan đến câu hỏi. Phạt điểm hoặc không cộng điểm cho các câu văn dài dòng, lặp lại hoặc chèn thông tin thừa không được hỏi."*
> 2. **Định nghĩa thang điểm 1–5 rõ ràng với penalty cho sự dài dòng:**
>    - **5/5:** Trả lời chính xác, đầy đủ ý cốt lõi, trình bày súc tích, ngắn gọn, không có từ ngữ thừa.
>    - **4/5:** Trả lời đúng và đủ ý nhưng dài dòng, có các đoạn giải thích không cần thiết.
> 3. **Yêu cầu Judge viết Rationale trước khi cho điểm:** Bắt buộc Judge phải liệt kê lý do cụ thể và phân tích tính ngắn gọn trước khi đưa ra điểm số cuối cùng (Chain-of-Thought prompting).

**Câu 3: Tại sao cần calibrate LLM judge với human labels?**

> *Câu trả lời:*
> Cần calibrate (hiệu chuẩn) LLM Judge với Human Labels vì các lý do sau:
> 
> 1. **Khắc phục Bias hệ thống của LLM:** LLM Judge thường mắc phải Position bias, Verbosity bias, Self-preference bias, và Style bias (thích định dạng Markdown rườm rà). Calibration giúp phát hiện và loại bỏ các lệch lạc này.
> 2. **Đảm bảo Alignment với chuyên môn tên miền (Domain Nuances):** Trong miền OrbitTech Customer Support, con người hiểu rõ các quy định kinh doanh, mức độ nghiêm trọng của sai sót (ví dụ: báo sai giá là lỗi nghiêm trọng, sai lỗi chính tả nhẹ là lỗi nhỏ). LLM Judge nếu không calibrated sẽ chấm điểm cào bằng hoặc đánh giá sai ưu tiên kinh doanh.
> 3. **Tính toán chỉ số tin cậy (Agreement Rate):** Giúp tính chỉ số đồng thuận Cohen's Kappa / Krippendorff's Alpha giữa LLM Judge và Chuyên gia con người. Chỉ khi đạt chỉ số tin cậy cao ($\ge 0.80$), LLM Judge mới đủ điều kiện thay thế con người trong các đợt offline evaluation quy mô lớn.

### Exercise 1.3 — Evaluation trong CI/CD

**Câu 1: Chọn threshold để block deployment.**

| Metric | Threshold | Lý do |
|---|---:|---|
| Faithfulness | **0.85** | Ngăn chặn rủi ro bịa đặt thông tin (hallucination) làm ảnh hưởng uy tín thương hiệu và nghĩa vụ pháp lý của OrbitTech. Nếu dưới 0.85, nguy cơ trả lời sai chính sách đổi trả/bảo hành là rất cao. |
| Answer Relevance | **0.80** | Đảm bảo câu trả lời trực diện giải quyết đúng thắc mắc của khách hàng, tránh trải nghiệm khó chịu khi bot trả lời lan man hoặc sai lệch chủ đề. |
| Completeness | **0.75** | Đảm bảo các quy trình/điều kiện quan trọng không bị bỏ sót. Ngưỡng 0.75 cho phép linh hoạt nhẹ về văn phong tóm tắt nhưng vẫn giữ đầy đủ các ý chính. |

**Câu 2: Khi nào dùng offline evaluation, online evaluation và human review?**

> *Câu trả lời:*
> 1. **Offline Evaluation:**
>    - **Khi nào dùng:** Thực hiện tự động trong quy trình CI/CD trước khi release phiên bản mới, khi thay đổi System Prompt, nâng cấp Embedding Model, thay đổi thuật toán Chunking hoặc đổi LLM Provider.
>    - **Mục đích:** Chạy trên tập Golden Dataset cố định với các chỉ số RAGAS và LLM Judge để phát hiện sớm lỗi regression (suy giảm chất lượng) trước khi code được merge/deploy.
> 
> 2. **Online Evaluation:**
>    - **Khi nào dùng:** Thực hiện liên tục trên môi trường Production với lưu lượng người dùng thật (real-time traffic).
>    - **Mục đích:** Theo dõi các chỉ số thời gian thực (latency, cost/tokens, error rate, user feedback like/dislike, implicit feedback như hỏi lại câu thứ 2). Phát hiện rủi ro trôi dạt dữ liệu (data drift) và các edge cases phát sinh trong thực tế mà Golden Dataset chưa bao phủ.
> 
> 3. **Human Review:**
>    - **Khi nào dùng:** Dùng khi xây dựng/duyệt tập Golden Dataset ban đầu, audit định kỳ ngẫu nhiên (5–10% dữ liệu production), kiểm tra các ca bị người dùng dislike/khiếu nại, và khi calibrate LLM Judge.
>    - **Mục đích:** Cung cấp "Ground Truth" chuẩn xác nhất, giải quyết các ca đánh giá phức tạp, tinh tế mà máy móc không tự quyết định được.

---

## Part 2 — Core Coding (14:45–15:40)

Hoàn thiện các TODO bắt buộc trong `template.py`.

### Task 1 — Data Models

- `QAPair`: question, expected answer, gold context, metadata và retrieved contexts.
- `EvalResult`: answer-side scores, optional retrieval scores, pass/failure fields.
- `overall_score()`: trung bình Faithfulness, Relevance và Completeness.

### Task 2 — RAGASEvaluator

Answer-side:

- `evaluate_faithfulness(answer, context)`
- `evaluate_relevance(answer, question)`
- `evaluate_completeness(answer, expected)`

Retrieval-side:

- `evaluate_context_recall(contexts, expected)`
- `evaluate_context_precision(contexts, expected)`

Full pipeline:

- `run_full_eval(..., contexts=None)` luôn tính ba answer metrics.
- Nếu có `contexts`, tính và lưu thêm Context Recall và Context Precision.
- Retrieval scores không làm thay đổi `overall_score()` và pass rule gốc.

### Task 3 — LLMJudge

- `score_response(question, answer, rubric)`
- `detect_bias(scores_batch)`

### Task 4 — BenchmarkRunner

- `run(qa_pairs, agent_fn, evaluator)`
- `generate_report(results)`
- `run_regression(new_results, baseline_results)`
- `identify_failures(results, threshold)`

`BenchmarkRunner.run()` phải truyền `pair.retrieved_contexts` vào
`run_full_eval()`. Report phải có average của hai retrieval metrics.

### Task 5 — FailureAnalyzer

- `categorize_failures(failures)`
- `find_root_cause(failure)`
- `generate_improvement_suggestions(failures)`
- `generate_improvement_log(failures, suggestions)`

Kiểm tra:

```bash
pytest tests/ -v
```

`rerank_by_overlap()` là TODO bonus của Exercise 3.5. Test tương ứng được skip
nếu bạn chưa làm bonus.

---

## Part 3 — Golden Dataset & Real Benchmark (15:40–16:35)

### Exercise 3.1 — Build the Golden Dataset

Thiết kế và validate dataset theo Mục 5–6 trong `guide_lab.md`. Nội dung 20 QA
được điền trực tiếp trong `golden_dataset.json`; phần dưới chỉ ghi lại kết quả
và quyết định thiết kế, không chép lại toàn bộ QA.

**Kết quả dataset**

| Hạng mục | Kết quả |
|---|---|
| Tổng số records | ____ / 20 |
| Easy | ____ / 5 |
| Medium | ____ / 7 |
| Hard | ____ / 5 |
| Adversarial | ____ / 3 |
| Source documents được sử dụng | ____ / 10 |
| Validator status | PASS / FAIL |

**Ba case đại diện cho quyết định thiết kế**

| ID | Difficulty | Source document(s) | Vì sao case phù hợp với difficulty/attack type? |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

**Điểm khó nhất khi xây dựng expected answer hoặc evidence là gì?**

> *Câu trả lời:*

**Xác nhận:**

- [ ] Mọi claim trong expected answer đều có evidence hỗ trợ.
- [ ] Không có questions trùng ý và không dùng kiến thức ngoài corpus.
- [ ] `python validate_golden_dataset.py` báo `PASS`.

### Exercise 3.2 — Benchmark Run

Chạy:

```bash
python domain_assistant.py
python evaluate_answers.py
```

Copy bảng terminal vào đây hoặc điền từ `artifacts/benchmark_results.json`.

| ID | Question (short) | Ctx Recall | Ctx Precision | Faithfulness | Relevance | Completeness | Overall | Passed? | Failure Type |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| E01 | | | | | | | | | |
| E02 | | | | | | | | | |
| E03 | | | | | | | | | |
| E04 | | | | | | | | | |
| E05 | | | | | | | | | |
| M01 | | | | | | | | | |
| M02 | | | | | | | | | |
| M03 | | | | | | | | | |
| M04 | | | | | | | | | |
| M05 | | | | | | | | | |
| M06 | | | | | | | | | |
| M07 | | | | | | | | | |
| H01 | | | | | | | | | |
| H02 | | | | | | | | | |
| H03 | | | | | | | | | |
| H04 | | | | | | | | | |
| H05 | | | | | | | | | |
| A01 | | | | | | | | | |
| A02 | | | | | | | | | |
| A03 | | | | | | | | | |

**Aggregate Report**

- Overall pass rate: ____%
- Avg Context Recall: ____
- Avg Context Precision: ____
- Avg Faithfulness: ____
- Avg Relevance: ____
- Avg Completeness: ____
- Failure type distribution: ____

**Ba cases có Overall Score thấp nhất**

1. ID: ____ | Score: ____ | Failure type: ____
2. ID: ____ | Score: ____ | Failure type: ____
3. ID: ____ | Score: ____ | Failure type: ____

**Nhận xét ngắn:** Metric nào yếu nhất? Kết quả gợi ý vấn đề nằm ở retrieval
hay generation?

> *Câu trả lời:*

### Exercise 3.3 — LLM-as-a-Judge Rubric Design

Thiết kế rubric domain-specific cho OrbitTech Customer Support. Mỗi mức phải
đủ cụ thể để hai người chấm độc lập có thể hiểu giống nhau.

Chọn 3–5 dimensions:

- [ ] Correctness
- [ ] Completeness
- [ ] Relevance
- [ ] Evidence/citation
- [ ] Actionability
- [ ] Safety/privacy
- [ ] Tone/clarity
- [ ] Dimension khác: __________

| Score | Tiêu chí domain-specific | Ví dụ response |
|---:|---|---|
| 5 | | |
| 4 | | |
| 3 | | |
| 2 | | |
| 1 | | |

**Ba edge cases khó chấm**

| Edge Case | Tại sao khó chấm? | Rubric xử lý thế nào? |
|---|---|---|
| | | |
| | | |
| | | |

**Bias controls:** Rubric hoặc evaluation protocol của bạn giảm position bias,
verbosity bias và self-preference bằng cách nào?

> *Câu trả lời:*

### Exercise 3.4 — Framework Comparison (Bonus +10)

Chỉ làm sau khi hoàn thành 3.1–3.3. Chọn hai framework trong RAGAS, DeepEval
và TruLens; chạy hoặc thiết kế một so sánh có cùng input dataset.

| Tiêu chí | Framework 1: ____ | Framework 2: ____ |
|---|---|---|
| Setup complexity | | |
| Metrics available | | |
| CI/CD integration | | |
| Kết quả trên cùng dataset | | |
| Insight rút ra | | |

- Scores có nhất quán không?
- Framework nào strict hơn và vì sao?
- Hai framework có tìm ra cùng failure cases không?

> *Phân tích:*

### Exercise 3.5 — Retrieval Reranking (Bonus +5)

Mục tiêu: kiểm tra việc đổi thứ tự chunks có tăng Context Precision mà không
thay đổi Context Recall hay không.

1. Chọn ít nhất 5 cases từ `artifacts/actual_answers.json`.
2. Tính Context Recall và Context Precision trước rerank.
3. Implement `rerank_by_overlap()` hoặc một reranker khác.
4. Rerank cùng tập chunks, không thêm hoặc xóa chunk.
5. Tính lại hai metrics và giải thích kết quả.

| ID | Recall before | Recall after | Precision before | Precision after | Delta Precision |
|---|---:|---:|---:|---:|---:|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| **Avg** | | | | | |

**Tại sao Recall dự kiến không đổi?**

> *Câu trả lời:*

**Khi nào reranking không đủ và cần sửa retriever/query/chunking?**

> *Câu trả lời:*

---

## Part 4 — Reflection (16:35–16:50)

Hoàn thành `reflection.md` bằng kết quả thật từ Exercise 3.2.

---

## Completion Checklist

Hoàn thành kiểm tra cuối trong khoảng 16:50–17:00.

- [ ] Tất cả required tests pass.
- [ ] `golden_dataset.json` validate thành công.
- [ ] Exercise 3.1 hoàn thành trong file JSON và bảng kết quả phía trên.
- [ ] Exercise 3.2 có năm metrics, aggregate report và ba cases thấp nhất.
- [ ] Exercise 3.3 có rubric 1–5 và bias controls.
- [ ] `reflection.md` có ba failure analyses và regression strategy.
- [ ] Đã copy `template.py` thành `solution/solution.py`.
- [ ] Exercise 3.4 và 3.5 chỉ làm nếu chọn bonus.

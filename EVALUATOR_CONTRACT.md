# HỢP ĐỒNG ĐÁNH GIÁ (EVALUATOR CONTRACT)
*Được kiểm chứng thực nghiệm 100% bằng probe suite `scripts/evaluator_probe.py` và `scripts/extended_evaluator_probe.py` trên `src/lifemate/evaluation.py`.*

Tài liệu này xác lập các luật toán học bất biến của bộ chấm điểm LifeMate. Mọi thiết kế agent, validator, và controller bắt buộc phải tuân thủ nghiêm ngặt các quy tắc dưới đây.

---

## 1. Bảng Kết quả Thực nghiệm (Probe Matrix)

| Test Case | Kịch bản Thử nghiệm | Kết quả Thực nghiệm | Ý nghĩa Kiến trúc & Hệ quả |
| :--- | :--- | :---: | :--- |
| **1. Order Permutation** | Hoán vị thứ tự các tool calls độc lập (`[A, B]` vs `[B, A]`) | **F1 = 1.0 (Matched 2/2)** | **Thứ tự tool calls độc lập không bị trừ điểm.** Controller chỉ cần đảm bảo Topological Order hợp lệ để đạt `state_hash` đúng. |
| **2a. Extra Key** | Truyền thừa 1 tham số không có trong nhãn (ví dụ `location`) | **F1 = 0.0 (Matched 0)** | **Bẫy `set(a) != set(b)`**: Thừa dù chỉ 1 key là mất trắng toàn bộ điểm của tool call đó! |
| **2b. Null Key** | Strict decoder ép tham số optional thành `null` | **F1 = 0.0 (Matched 0)** | **CẤM Strict Null Encoding**: Không được đưa key có value `None`/`null` vào arguments. |
| **3. Missing Optional** | Bỏ sót tham số mà Gold có truyền (ví dụ `end` time) | **F1 = 0.0 (Matched 0)** | **Thiếu key là 0 điểm**: Phải dùng `priors.py` từ 1.545 mẫu để biết tập tham số chuẩn tắc. |
| **4a. String vs Int** | Truyền `"500000"` (str) thay vì `500000` (int) | **F1 = 0.0 (Matched 0)** | **Type Sensitivity tuyệt đối**: 206 trường Exact Match đòi hỏi `type(pred) == type(gold)`. |
| **4b. Float vs Int** | Truyền `500000.0` (float) thay vì `500000` (int) | **F1 = 0.0 (Matched 0)** | Ép kiểu số nguyên bắt buộc phải là `int`, không để `float`. |
| **4c. Bool vs Int** | Truyền `1` thay cho `True` (boolean) | **F1 = 0.0 (Matched 0)** | `type(1) == type(True)` thất bại (`int != bool`). Không được cast bool thành 1/0. |
| **5. Extra Read Call** | Gọi thêm 1 tool đọc không cần thiết | **F1 = 0.67 (Prec=0.5)** | **Phạt Precision**: Cắt tỉa mọi lệnh đọc không phục vụ cho lệnh ghi cuối cùng. |
| **6. Out-of-Scope** | Gold là `[]`, Prediction là `[]` | **F1 = 1.0** | Khi không có app hỗ trợ, trả về `[]` lập tức để bảo toàn F1 = 1.0 tuyệt đối. |
| **7a. Fuzzy Boundary** | Tham số có `x-match: fuzzy` (Token F1 đúng bằng 0.600) | **F1 = 0.0 (False)** | **Strictly `> 0.6`**: Bắt buộc Token F1 phải lớn hơn hẳn 0.6, bằng 0.6 vẫn bị tính là sai! |
| **7b. Fuzzy Overlap** | Tham số có `x-match: fuzzy` (Token F1 > 0.6) | **F1 = 1.0 (True)** | Chỉ 40 trường tự do được phép chênh lệch từ vựng; 206 trường còn lại bắt buộc Exact Match. |
| **8a. Nested List** | Thứ tự phần tử trong mảng bị đảo (`["b", "a"]` vs `["a", "b"]`) | **F1 = 0.0 (False)** | **Thứ tự mảng là Strict**: Thứ tự phần tử trong list arguments bắt buộc phải khớp. |
| **8b. Nested Dict** | Thứ tự các key trong dictionary lồng nhau khác nhau | **F1 = 1.0 (True)** | Thứ tự key trong dictionary lồng nhau là bất biến (invariant). |
| **9. Duplicate Call** | Gọi lặp lại 2 lần cùng một tool và arguments | **F1 = 0.67 (Prec=0.5)** | **Anti-duplicate Ledger**: Bắt buộc chặn các lệnh gọi trùng lặp vô ý thức. |

---

## 2. Kết quả Kiểm toán Dữ liệu (Dataset Audit Facts)

1. **Tổng số mẫu Train thực tế**: Chính xác **1.545 mẫu** (tương ứng chính xác 1.545 thư mục âm thanh trong `train/audio`). Các ID trải dài từ `train_0001` đến `train_1753` do Ban tổ chức bỏ qua ~208 số thứ tự trong quá trình tạo dataset.
2. **Kiểm toán Giá trị `null` trong Gold**:
   - Tổng số tool calls quét qua: **3.541 calls**.
   - Tổng số arguments quét qua: **4.565 arguments**.
   - **Số lượng argument mang giá trị `null` trong nhãn Gold: CHÍNH XÁC 0!**
   - **Kết luận**: Nhãn của Ban tổ chức **KHÔNG BAO GIỜ** chứa giá trị `None` / `null`. Mọi tham số không sử dụng đều bị lược bỏ hoàn toàn khỏi key set. Do đó, việc xóa các key `None` trong `safe_canonicalize` là hoàn toàn chính xác 100%.

---

## 3. Các Ràng buộc Bắt buộc đối với Pipeline

1. **Về Arguments**:
   - `set(pred_args) == set(gold_args)` là điều kiện tiên quyết.
   - Không emit key có giá trị `None` / `null`.
   - Lossless Cast: Chuỗi số nguyên $\rightarrow$ `int` khi schema type là `integer`.
   - Không convert `1` thành `True` cho boolean.
2. **Về Termination**:
   - Không được dùng `no tool_call == DONE`. Phải xác nhận toàn bộ mục tiêu đã thỏa mãn (`all_goals_satisfied`) mới cho phép dừng.
3. **Về Tool Selection**:
   - Tối thiểu hóa số lượng calls (`minimum sufficient trajectory`). Không gọi thêm tool đọc suy đoán nếu context/observations đã có dữ liệu.

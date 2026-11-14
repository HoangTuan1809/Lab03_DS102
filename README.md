# Lab 03 - DS102: Linear Regression (Forest Fires Dataset)

Xây dựng mô hình **Linear Regression** từ đầu bằng NumPy thuần, thực hiện tiền xử lý dữ liệu, xử lý đa cộng tuyến và đối chiếu với `scikit-learn` trên bộ dữ liệu **Forest Fires**.

## Mô tả chi tiết các file & Kết quả

### 1. `Linear_Regression.py`
* **Nội dung:** Xây dựng class `LinearRegression` bằng thư viện NumPy thuần, tối ưu tham số theo phương pháp Normal Equation kết hợp giả nghịch đảo Moore-Penrose (`np.linalg.pinv`), tích hợp sẵn các phương thức `fit`, `predict` và `evaluate` (MSE, RMSE, R^2).
* **Vai trò:** Module cốt lõi được import và tái sử dụng xuyên suốt 4 bài tập.

### 2. `Assignment_01.py`
* **Nội dung:** Nạp bộ dữ liệu `forestfires.csv`, mã hóa One-Hot cho hai cột phân loại `day` và `month`, chia tập Train/Test (80/20) và huấn luyện mô hình `LinearRegression` tự code trên dữ liệu gốc.
* **Kết quả:**
  * **Tập Train:** `MSE = 1,976.09` | `RMSE = 44.45` | `R2 = 0.05`
  * **Tập Test:** `MSE = 11,613.67` | `RMSE = 107.77` | `R2 = 0.01`
  * *Nhận xét:* Biến mục tiêu `area` bị lệch phải nặng làm kéo lệch đường hồi quy tuyến tính.

### 3. `Assignment_02.py`
* **Nội dung:** Chuẩn hóa Z-score toàn bộ đặc trưng đầu vào `X` (tính tham số trên tập Train để tránh Data Leakage), sau đó huấn luyện lại mô hình và đối chiếu với dữ liệu gốc ở Assignment 1.
* **Kết quả:**
  * **Tập Train:** `MSE = 1,976.09` | `RMSE = 44.45` | `R2 = 0.05`
  * **Tập Test:** `MSE = 11,613.67` | `RMSE = 107.77` | `R2 = 0.01`
  * *Nhận xét:* Với nghiệm giải tích Normal Equation, phép co giãn tuyến tính không làm thay đổi giá trị dự đoán và sai số, nhưng đưa các đặc trưng về cùng thang đo để so sánh trực tiếp độ lớn trọng số.

### 4. `Assignment_03.py`
* **Nội dung:** Chẩn đoán hiện tượng đa cộng tuyến trên tập Train bằng ma trận tương quan Pearson và chỉ số VIF. Phát hiện cặp `DC` – `DMC` có tương quan cao (r = 0.6831), tiến hành loại bỏ `DC` và chuẩn hóa Z-score cho `X` (giữ nguyên thang đo gốc của `area`) để đối chiếu trực tiếp giữa dữ liệu gốc (`Original`) và dữ liệu đã tiền xử lý (`Preprocessed`).
* **Kết quả:**
  * **Original Data:** Train `MSE = 1,976.09` (`RMSE = 44.45`, `R2 = 0.0498`) | Test `MSE = 11,613.67` (`RMSE = 107.77`, `R2 = 0.0148`)
  * **Preprocessed Data:** Train `MSE = 1,990.93` (`RMSE = 44.62`, `R2 = 0.0427`) | Test `MSE = 11,716.82` (`RMSE = 108.24`, `R2 = 0.0060`)
  * *Nhận xét:* Việc loại bỏ biến cộng tuyến `DC` giúp mô hình gọn hơn, giảm dư thừa thông tin mà vẫn bảo toàn gần như nguyên vẹn độ chính xác so với dữ liệu gốc; tuy nhiên R^2 vẫn thấp do biến `area` gốc bị lệch phải nặng và chứa các điểm ngoại lai lớn.

### 5. `Assignment_04.py`
* **Nội dung:** Triển khai mô hình hồi quy tuyến tính bằng thư viện `sklearn.linear_model.LinearRegression` và đối chiếu trực tiếp trọng số, bias cùng các chỉ số đánh giá (MSE, RMSE, R^2) với mô hình tự xây dựng từ `Linear_Regression.py` trên cả hai bộ dữ liệu gốc và đã tiền xử lý.
* **Kết quả:** Hai mô hình cho kết quả trùng khớp hoàn toàn (độ sai lệch trọng số và giá trị dự đoán chỉ ở mức 10^-13 do sai số dấu phẩy động).

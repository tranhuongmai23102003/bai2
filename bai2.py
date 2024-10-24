# Import các thư viện cần thiết
import cv2  # Thư viện OpenCV để xử lý ảnh
import numpy as np  # Thư viện NumPy để thao tác mảng
import matplotlib.pyplot as plt  # Thư viện matplotlib để hiển thị hình ảnh

# Đường dẫn đến file ảnh
image_path = r'C:\Users\tranm\Downloads\th.jpg'

# 1. Đọc ảnh gốc với chế độ ảnh xám (Grayscale)
# cv2.imread() dùng để đọc ảnh từ file, tham số cv2.IMREAD_GRAYSCALE cho biết chỉ đọc kênh xám
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 2. Dò biên với toán tử Sobel
# Tạo kernel Sobel cho hướng x (phát hiện cạnh theo chiều ngang)
# np.array tạo ra ma trận kernel 3x3 để tính toán gradient dọc theo trục x
sobel_x = cv2.filter2D(image, cv2.CV_64F, np.array([[-1, 0, 1], 
                                                    [-2, 0, 2], 
                                                    [-1, 0, 1]]))

# Tạo kernel Sobel cho hướng y (phát hiện cạnh theo chiều dọc)
# np.array tạo ra ma trận kernel 3x3 để tính toán gradient dọc theo trục y
sobel_y = cv2.filter2D(image, cv2.CV_64F, np.array([[-1, -2, -1], 
                                                    [0,  0,  0], 
                                                    [1,  2,  1]]))

# Kết hợp hai gradient để tính độ lớn biên tổng hợp
# cv2.magnitude() sẽ tính căn bậc hai của tổng bình phương gradient theo cả hai trục (x và y)
sobel_combined = cv2.magnitude(sobel_x, sobel_y)

# 3. Dò biên với toán tử Laplace Gaussian (LoG)
# Tạo kernel Laplacian of Gaussian (LoG) để phát hiện biên
# Đây là ma trận 5x5 biểu diễn bộ lọc LoG
log_kernel = np.array([[0,  0, -1,  0,  0],
                       [0, -1, -2, -1,  0],
                       [-1, -2, 16, -2, -1],
                       [0, -1, -2, -1,  0],
                       [0,  0, -1,  0,  0]])

# Áp dụng kernel LoG lên ảnh xám để phát hiện biên
# cv2.filter2D() áp dụng bộ lọc convolution 2D với ảnh gốc và kernel LoG
log_image = cv2.filter2D(image, -1, log_kernel)

# 4. Hiển thị kết quả các bước xử lý
plt.figure(figsize=(10, 5))  # Tạo một khung hình với kích thước 10x5 inch

# Hiển thị ảnh gốc xám
plt.subplot(1, 3, 1)  # Đặt hình ảnh vào vị trí đầu tiên trong grid 1x3
plt.imshow(image, cmap='gray')  # Hiển thị ảnh dưới dạng grayscale
plt.title('Ảnh xám')  # Tiêu đề cho hình ảnh

# Hiển thị kết quả dò biên với toán tử Sobel
plt.subplot(1, 3, 2)  # Đặt hình ảnh vào vị trí thứ hai
plt.imshow(sobel_combined, cmap='gray')  # Hiển thị ảnh biên với Sobel
plt.title('Biên với Sobel')  # Tiêu đề

# Hiển thị kết quả dò biên với toán tử LoG
plt.subplot(1, 3, 3)  # Đặt hình ảnh vào vị trí thứ ba
plt.imshow(log_image, cmap='gray')  # Hiển thị ảnh biên với LoG
plt.title('Biên với LoG')  # Tiêu đề

# Hiển thị tất cả các hình ảnh đã đặt với bố cục gọn gàng
plt.tight_layout()  # Điều chỉnh layout để tránh trùng lặp
plt.show()  # Hiển thị các hình ảnh lên màn hình

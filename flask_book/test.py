import pandas as pd

# Đọc file CSV
df = pd.read_csv('/media/anh/428916C82C800CE5/langchain_final/flask_book/book_genre.csv')

# Kiểm tra dữ liệu
print(df.head())  # In ra 5 dòng đầu tiên để xác nhận dữ liệu

# Tìm kiếm tiêu đề
def find_title_by_summary(summary_to_find):
    # Lọc các dòng có summary phù hợp
    result = df[df['summary'].str.contains(summary_to_find, na=False)]
    
    if not result.empty:
        return result['title'].tolist()  # Trả về danh sách tiêu đề
    else:
        return None

# Ví dụ: Tìm kiếm bằng summary
summary_to_search = "Trong quá trình trẻ trưởng thành những khi cha mẹ nặng lời"
titles_found = find_title_by_summary(summary_to_search)

if titles_found:
    for title in titles_found:
        print(f"Tiêu đề tìm thấy: {title}")
else:
    print("Không tìm thấy tiêu đề nào khớp.")

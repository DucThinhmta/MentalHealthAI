# Dữ liệu (không đưa lên repo)

Tải về và đặt vào thư mục `data/`, rồi chạy `python src/clean_data.py`.

| Nhánh | Bộ dữ liệu | Nguồn |
|---|---|---|
| Stress | Dreaddit (Turcan & McKeown, 2019) | https://aclanthology.org/D19-6213/ |
| Emotion | GoEmotions (Demszky et al., 2020) | https://github.com/google-research/google-research/tree/master/goemotions |
| Depression | Depression Severity Dataset (Naseem et al., 2022) | https://github.com/usmaann/Depression_Severity_Dataset |

Tên tệp mà code mong đợi:
- Stress: `train_clean.csv`, `test_clean.csv` (cột `text_clean`, `label`)
- Emotion: `train_emotion.csv`, `dev_emotion.csv`, `test_emotion.csv` (cột `text`, `label_vector`)
- Depression: `depression_train.csv`, `depression_test.csv` (cột `text`, `label` 0-3)

Hãy đọc điều khoản sử dụng của từng bộ dữ liệu trước khi chia sẻ lại.

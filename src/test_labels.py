import pandas as pd


def convert_labels(label_text):
    vector = [0] * 28

    labels = label_text.split(",")

    for label in labels:
        label_id = int(label)
        vector[label_id] = 1

    return vector


def process_dataset(input_file, output_file):
    # Đọc file TSV
    df = pd.read_csv(
        input_file,
        sep="\t",
        header=None,
        names=["text", "labels", "id"]
    )

    # Chuyển nhãn
    df["label_vector"] = df["labels"].apply(convert_labels)

    # Chỉ giữ text và vector nhãn
    result = df[["text", "label_vector"]].copy()

    # Lưu CSV
    result.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig"
    )

    print("Đã xử lý:", input_file)
    print("Số dòng:", len(result))


# Train
process_dataset(
    "./MentalHealthAI/data/train.tsv",
    "./MentalHealthAI/data/train_emotion.csv"
)

# Dev
process_dataset(
    "./MentalHealthAI/data/dev.tsv",
    "./MentalHealthAI/data/dev_emotion.csv"
)

# Test
process_dataset(
    "./MentalHealthAI/data/test.tsv",
    "./MentalHealthAI/data/test_emotion.csv"
)
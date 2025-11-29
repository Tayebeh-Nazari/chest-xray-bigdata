import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit

def create_spark_session():
    """
    Creates and returns a Spark session.
    """
    spark = (
        SparkSession.builder
        .appName("XRay-Ingestion")
        .master("local[*]")  # remove in cluster mode
        .getOrCreate()
    )
    return spark


def load_image_paths(base_dir):
    """
    Given a base directory (e.g. data/raw), returns a list of tuples:
    (image_path, label)
    
    Directory structure must look like:
        data/raw/
            Pneumonia/
            No_Finding/
            Effusion/
    """
    data = []
    for label in os.listdir(base_dir):
        class_path = os.path.join(base_dir, label)
        
        # skip files accidentally thrown in raw/
        if not os.path.isdir(class_path):
            continue
        
        for file_name in os.listdir(class_path):
            if file_name.lower().endswith((".png", ".jpg", ".jpeg")):
                full_path = os.path.join(class_path, file_name)
                data.append((full_path, label))
    
    return data


def to_spark_dataframe(spark, data):
    """
    Converts list[(path, label)] into a Spark DataFrame.
    """
    df = spark.createDataFrame(data, ["image_path", "label"])
    return df


def main():
    BASE_DIR = "data/raw"

    print("🔵 Starting ingestion...")
    print(f"📁 Reading from: {BASE_DIR}")

    # 1. Create spark session
    spark = create_spark_session()

    # 2. Collect image paths and labels
    data = load_image_paths(BASE_DIR)
    print(f"📸 Found {len(data)} images.")

    # 3. Convert to Spark DataFrame
    df = to_spark_dataframe(spark, data)

    print("🧪 Showing sample rows:")
    df.show(5, truncate=False)

    print("📊 Total images in Spark DataFrame:", df.count())

    # Keep session alive for manual inspection
    spark.stop()


if __name__ == "__main__":
    main()


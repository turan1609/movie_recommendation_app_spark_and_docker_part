print(">>> CHECKPOINT 0: Kütüphaneler yükleniyor ve Spark başlatılıyor...")

import time
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, udf, collect_list
from pyspark.sql.types import IntegerType, FloatType, StringType, StructType, StructField
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.feature import Normalizer

home_dir = os.path.expanduser('~')
spark_temp_dir = os.path.join(home_dir, "spark-temp")
spark_checkpoint_dir = os.path.join(home_dir, "spark-checkpoints")

os.makedirs(spark_temp_dir, exist_ok=True)
os.makedirs(spark_checkpoint_dir, exist_ok=True)

spark = (SparkSession.builder
    .appName("ALS_Model_Optimization")
    .config("spark.driver.memory", "16g")
    .config("spark.executor.memory", "16g")
    .config("spark.sql.shuffle.partitions", "200")
    .config("spark.local.dir", spark_temp_dir)
    .master("local[*]")
    .getOrCreate()
)

sc = spark.sparkContext
sc.setLogLevel("WARN")
sc.setCheckpointDir(spark_checkpoint_dir)

print(f"SparkSession başlatıldı. Spark Sürümü: {spark.version}")
print(f"Geçici Spark dosyaları için kullanılacak dizin: {spark_temp_dir}")
print(f"Checkpoint dizini: {spark_checkpoint_dir}")



print("\n>>> CHECKPOINT 1: Veriler yükleniyor...")

BASE_PATH = "./ml-latest/"
MOVIES_PATH = f"{BASE_PATH}movies.csv"
RATINGS_PATH = f"{BASE_PATH}ratings.csv"
GENOME_SCORES_PATH = f"{BASE_PATH}genome-scores.csv"

movie_schema = StructType([
    StructField("movieId", IntegerType(), True),
    StructField("title", StringType(), True),
    StructField("genres", StringType(), True)
])

rating_schema = StructType([
    StructField("userId", IntegerType(), True),
    StructField("movieId", IntegerType(), True),
    StructField("rating", FloatType(), True),
    StructField("timestamp", IntegerType(), True)
])

genome_scores_schema = StructType([
    StructField("movieId", IntegerType(), True),
    StructField("tagId", IntegerType(), True),
    StructField("relevance", FloatType(), True)
])

movies_df = spark.read.csv(MOVIES_PATH, header=True, schema=movie_schema, escape='"')
ratings_df_raw = spark.read.csv(RATINGS_PATH, header=True, schema=rating_schema)
genome_scores_df = spark.read.csv(GENOME_SCORES_PATH, header=True, schema=genome_scores_schema)

movies_df.cache()
ratings_df_raw.cache()
genome_scores_df.cache()

print(f"  - Filmler yüklendi: {movies_df.count()} adet")
print(f"  - Derecelendirmeler yüklendi: {ratings_df_raw.count()} adet")
print(f"  - Genom skorları yüklendi: {genome_scores_df.count()} adet")

print("\n>>> CHECKPOINT 2: Veri ön işleme adımları...")

als_ratings_df = ratings_df_raw.select(
    col("userId").alias("user"),
    col("movieId").alias("item"),
    "rating"
).cache()
print("  - ALS için veri hazırlığı tamamlandı.")

list_to_vector_udf = udf(lambda l: Vectors.dense(sorted(l)), VectorUDT())

movies_tag_genome_df = (genome_scores_df
                        .groupBy("movieId")
                        .agg(collect_list("relevance").alias("relevance_list"))
                        .orderBy("movieId"))

movies_featured_df_raw = movies_tag_genome_df.withColumn(
    "features",
    list_to_vector_udf(col("relevance_list"))
).select("movieId", "features")

normalizer = Normalizer(inputCol="features", outputCol="normFeatures")
movies_featured_df = normalizer.transform(movies_featured_df_raw).selectExpr("movieId", "normFeatures as features").cache()

print("  - Kosinüs benzerliği için 'Tag Genome' tabanlı özellik vektörleri oluşturuldu.")

def find_similar_movies(movie_id, top_n=10):
    try:
        target_vector_row = movies_featured_df.filter(col("movieId") == movie_id).first()
        if not target_vector_row:
            print(f"Uyarı: {movie_id} ID'li film için özellik vektörü bulunamadı.")
            return None
        target_vector = target_vector_row.features
        
        dot_product_udf = udf(lambda x: float(x.dot(target_vector)), FloatType())
        
        similarities_df = movies_featured_df.withColumn("similarity", dot_product_udf(col("features")))
        
        top_similar_movies = (similarities_df
                              .filter(col("movieId") != movie_id)
                              .orderBy(col("similarity").desc())
                              .limit(top_n)
                              .join(movies_df, "movieId"))
        return top_similar_movies
    except Exception as e:
        print(f"Benzer filmler bulunurken hata: {e}")
        return None

print("  - 'find_similar_movies' fonksiyonu tanımlandı.")

print("\n>>> CHECKPOINT 3: ALS modeli için hiperparametre optimizasyonu...")

(training_df, test_df) = als_ratings_df.randomSplit([0.8, 0.2], seed=42)
training_df.cache()
test_df.cache()
print(f"  - Eğitim seti boyutu: {training_df.count()}")
print(f"  - Test seti boyutu: {test_df.count()}")

als = ALS(
    userCol="user",
    itemCol="item",
    ratingCol="rating",
    coldStartStrategy="drop",
    nonnegative=True
)

param_grid = (ParamGridBuilder()
              .addGrid(als.rank, [12, 20, 200])
              .addGrid(als.regParam, [0.1, 0.5, 1.0])
              .addGrid(als.maxIter, [15, 20, 200])
              .build()
             )

evaluator = RegressionEvaluator(
    metricName="rmse",
    labelCol="rating",
    predictionCol="prediction"
)

cross_validator = CrossValidator(
    estimator=als,
    estimatorParamMaps=param_grid,
    evaluator=evaluator,
    numFolds=3,
    parallelism=4,
    seed=42
)

print(f"\n  - Optimizasyon başlıyor. Toplam {len(param_grid)} model denenecek...")
print("  - BU İŞLEM UZUN SÜRECEKTİR...")

start_time = time.time()
cv_model = cross_validator.fit(training_df)
end_time = time.time()

print(f"\n  - Optimizasyon ve eğitim { (end_time - start_time) / 60:.2f} dakikada tamamlandı.")

als_model = cv_model.bestModel

print("\n--- EN İYİ MODEL SONUÇLARI ---")
print(f"  - En İyi Rank: {als_model.rank}")
print(f"  - En İyi MaxIter: {als_model.getMaxIter()}")
print(f"  - En İyi RegParam: {als_model.getRegParam():.4f}")

print("\n  - En iyi modelin final performansı değerlendiriliyor...")

train_predictions = als_model.transform(training_df)
train_rmse = evaluator.evaluate(train_predictions)
print(f"  - Eğitim Seti RMSE: {train_rmse:.4f}")

test_predictions = als_model.transform(test_df)
test_rmse = evaluator.evaluate(test_predictions)
print(f"  - Test Seti RMSE: {test_rmse:.4f}")

overfitting_ratio = (test_rmse - train_rmse) / train_rmse * 100
print(f"\n  - Test hatası, eğitim hatasından %{overfitting_ratio:.2f} daha yüksek.")
if overfitting_ratio > 10:
    print("  - DİKKAT: Modelde overfitting (aşırı öğrenme) eğilimi olabilir.")
else:
    print("  - Modelin genelleme performansı iyi görünüyor.")

print("\n>>> MODEL OPTİMİZASYONU TAMAMLANDI. 'als_model' değişkeni en iyi modeli içermektedir.")

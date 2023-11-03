import tensorflow as tf
from tensorflow import keras
import pandas as pd
import os
import time
import sys
from tensorflow.train import BytesList
from tensorflow.train import Feature, Features, Example

"""원본 cookbook recipe는 https://www.kaggle.com/datasets/paultimothymooney/recipenlg에서 다운로드 가능"""

"""본 문서의 목표는 현재 다운로드되어있는 cookbook_recipes.csv 파일을
여러 csv파일로 쪼갠 뒤, 해당 파일들을 순회하여 데이터를 불러오는 
데이터셋을 만드는 것이다."""

# 1. cookbook recipe를 불러오고, 파일을 어떻게 쪼갤 지 결정하기
def read_dataframe(frame_path):
    df = pd.read_csv(frame_path)
    df = df.iloc[:, 2:]
    print(df.info())
    print(df.head())
    return df


# 2. cookbook recipe가 쪼개져 들어갈 폴더 생성하고 거기에 레시피를 쪼개서 저장
def split_dataframe(df, split_count=10):
    split_size = len(df) // split_count
    for i in range(1, split_count+1):
        os.makedirs(f"cookbook/{i}", exist_ok=True)
        try:
            locals()[f'df{i}'] = df[split_size * (i-1) : split_size * (i)]
            locals()[f'df{i}'].to_csv(f"cookbook/{i}/cookbook_{i}.csv")
        except:
            locals()[f'df{i}'] = df[split_size * i :]
            locals()[f'df{i}'].to_csv(f"cookbook/{i}/cookbook_{i}.csv")


# 3. 쪼개진 폴더에서 파일을 찾아 경로를 담는다
def filepaths(data):
    file_list = []
    for parent, _, files in os.walk(f'{data}'):
        for f in files:
            file_list.append(os.path.join(parent, f))
    return file_list


# 4. 위 filepaths에서 만들어진 파일경로 리스트에서 파일목록을 담고 있는 데이터셋을 가져온다
def filepath_dataset(file_list, seed=42):
    file_dataset = tf.data.Dataset.list_files(file_list, seed=seed)
    return file_dataset


# 5. 파일목록 데이터셋을 바탕으로 interleave를 사용해 데이터를 번갈아가며 읽는다
"""일반적으로 맨 윗줄은 헤더이기 때문에 두 번째 줄부터 읽는다"""
def read_data(file_dataset, n_readers=5):
    dataset = file_dataset.interleave(lambda file: tf.data.TextLineDataset(file).skip(1), cycle_length=n_readers,
                                      num_parallel_calls=tf.data.AUTOTUNE)
    return dataset


# cookbook = read_dataframe('cookbook_recipes.csv')
# split_dataframe(cookbook, 10)

# 6. 데이터셋으로 적재하기 vs CSV파일 통으로 가져오기 - 시간과 메모리 사용량 비교

start = time.perf_counter()
print(filepaths('cookbook'))
file_list = filepaths('cookbook')
print(filepath_dataset(file_list, 42))
file_dataset = filepath_dataset(file_list, 42)
dataset = read_data(file_dataset, 5)
for line in dataset.take(5):
    print(line.numpy())
end = time.perf_counter()
elapsed = end - start
print('5줄 호출에 걸린 시간(초)(from Dataset) :', elapsed)
print('메모리 할당량 :', sys.getsizeof(dataset))

start = time.perf_counter()
cookbooks = pd.read_csv('cookbook_recipes.csv')
print(cookbooks.head(5))
end = time.perf_counter()
elapsed = end - start
print('5줄 호출에 걸린 시간(초)(from CSV, pandas) :', elapsed)
print('메모리 할당량 :', sys.getsizeof(cookbooks))

start = time.perf_counter()
f = open('cookbook_recipes.csv', 'r')
fileread = f.readlines()[:6]
print(fileread)
end = time.perf_counter()
elapsed = end - start
print('5줄 호출에 걸린 시간(초)(from read) :', elapsed)
print('메모리 할당량 :', sys.getsizeof(fileread))

"""왜 데이터셋을 만들어야 하는가?
그 결과는 주어진 결과에 잘 나타나 있다. 
1. 호출 시간 - 해당 CSV파일(약 577MB)에서 5개의 줄을 가져온다고 생각하자. tf.data.Dataset 객체에서 병렬처리하여 5줄을 불러오는데 걸리는 시간과, 
해당 CSV 파일을 통째로 불러와서 5줄을 가져오는데 걸리는 시간의 차이는 거의 7배에 달한다. 실제 업무에서는 그보다 훨씬 큰 파일을 다루게 될 것이므로
그 차이는 더욱 커질 것이다. 그렇다고 그냥 csv를 텍스트로 불러오는 것은 미리 전처리 함수를 짜놓지 않는 이상 알아보기도 어렵고, 데이터셋보다 시간도 오래 걸린다.
(심지어 매번 형식이 달라질 것이므로 함수 자체의 호환성도 갖추기 어렵다)

그 차이는 메모리 할당량에서 더 크게 드러난다. Dataset 객체 하나는 48바이트를 차지했다. 그 반면 CSV는 당연히 원본을 통째로 적재하므로 할당량이 어마어마하다.
결국 원본을 통째로 메모리에 적재하는 것은 어느 순간 불가능해지므로 데이터셋 생성은 선택이 아니라 필수인 것이다."""

# 7. 굳이 TFRecord를 사용해야 하는가?
"""동일한 CSV를 TFRecord로 만들어서 호출하면 시간이 더 짧아지는가?"""
# cookbook = cookbooks.iloc[:, 2:]
# cookbook.to_csv('cookbooks_CleanHeader.csv')

def convert_csv_to_tfrecord(filename:str):
    "filename.csv to filename.tfrecord"
    df = pd.read_csv(f'{filename}.csv', index_col=0)
    HEADER = df.columns.values
    HEADER_DEFAULTS = [['NA'] for _ in HEADER]

    df_bytes = df.apply(lambda col: col.astype(str).str.encode('utf-8'))

    with tf.io.TFRecordWriter(f'{filename}.tfrecord') as writer:
        for i in range(len(df)):
            example = Example(
                features=Features(
                    feature={
                        header: Feature(bytes_list=BytesList(value=[df_bytes[header][i]]))
                        for header in HEADER
                    }
                )
            )
            writer.write(example.SerializeToString())


# convert_csv_to_tfrecord('cookbooks_CleanHeader')

"""tfrecord와 일반 데이터형식(csv)의 하드웨어적 비교

우선 데이터가 하드디스크 내에서 차지하는 용량을 비교해보면 tfrecord가 csv보다 크다 (직접 확인해보는 것은 어렵지 않다)
그럼에도 불구하고 tfrecord를 사용하는 이유는?"""

# 7-1. TFRecord의 압축?

def convert_csv_to_tfrecord_with_compression(filename:str):
    "same function as above but with compression"
    options = tf.io.TFRecordOptions(compression_type='GZIP')
    df = pd.read_csv(f'{filename}.csv', index_col=0)
    HEADER = df.columns.values
    HEADER_DEFAULTS = [['NA'] for _ in HEADER]

    df_bytes = df.apply(lambda col: col.astype(str).str.encode('utf-8'))

    with tf.io.TFRecordWriter(f'{filename}.tfrecord', options) as writer:
        for i in range(len(df)):
            example = Example(
                features=Features(
                    feature={
                        header: Feature(bytes_list=BytesList(value=[df_bytes[header][i]]))
                        for header in HEADER
                    }
                )
            )
            writer.write(example.SerializeToString())

# convert_csv_to_tfrecord_with_compression('cookbooks_CleanHeader')

"""GZIP 타입으로 압축하고 나니 용량이 대폭 줄어들었다. 압축하지 않은 tfrecord가 csv보다 100MB 이상 컸는데, 
압축된 tfrecord는 400MB나 더 작다. 주의할 점은, 따로 압축파일을 만드는 것이 아니라 tfrecord내에 압축되어 기록되었다는 점이다"""

# dataset = tf.data.TFRecordDataset('cookbooks_CleanHeader.tfrecord', compression_type='GZIP')
# for item in dataset.take(5):
#     print(item)

"""주의할 점은, 압축된 TFRecord 파일을 읽으려면 압축형식을 지정해줘야 한다. (미지정시 corrupted되었다는 에러가 발생)
 따라서 만약 TFRecord를 서버로 전송한다면,
만약 압축이 되었다면 압축 형식을 같이 전달하거나 읽는 쪽에서 예외를 건너뛰며 compression_type을 순회하는 함수를 만들어야할 것이다.
(다행히 GZIP, ZLIP, None 밖에 없다)

그렇다면 이 압축된 TFRecord를 읽는 것과 앞서 데이터를 나눈 뒤 순회하는 방식과 어떤 것이 더 빠를까?"""

# 7-2.
start = time.perf_counter()
dataset = tf.data.TFRecordDataset('cookbooks_CleanHeader.tfrecord', compression_type='GZIP')
for item in dataset.take(5):
    print(item)
end = time.perf_counter()
elapsed = end - start
print('5줄 호출에 걸린 시간(초)(from tfrecord) :', elapsed)
print('메모리 할당량 :', sys.getsizeof(dataset))

"""두 방식의 메모리 할당량은 신기하게도 같다. (정확히는 48Bytes) 하지만 속도에 있어서는 tfrecord에서 바로 불러오는 것이 조금 더 앞서는 경향을 보인다."""

# 8. tfrecord 파일 내에서 데이터를 나누고 이리저리 만져보기

dataset = (tf.data.TFRecordDataset('cookbooks_CleanHeader.tfrecord', compression_type='GZIP', num_parallel_reads=tf.data.AUTOTUNE)
           .shuffle(20000)
           .batch(32)
           .prefetch(tf.data.AUTOTUNE))

for item in dataset.take(1):
    print(item[:3])

"""주의할 점은 dataset을 배치단위로 묶음으로써, take 메서드에 포함되는 하나의 단위는 batch_size만큼의 크기를 가진다.
따라서 굳이 배치사이즈 별로 불러온 다음 개별 item을 찾고 싶다면, batch_size의 item(배치)의 개수를 추가로 지정해줘야 한다
(배치사이즈가 없었을 때는 take는 개별 단위의 item을 가져왔다)"""

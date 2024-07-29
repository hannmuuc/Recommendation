import pandas as pd

# 创建一个示例DataFrame
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Age': [25, 30, 35, 20],
    'Score': [88, 92, 76, 85]
}
df = pd.DataFrame(data)

# 根据单个列进行排序
# 升序排序
df_sorted_asc = df.sort_values(by='Age')
print("升序排序（按Age）:")
print(df_sorted_asc)

# 降序排序
df_sorted_desc = df.sort_values(by='Age', ascending=False)
print("\\n降序排序（按Age）:")
print(df_sorted_desc)

# 根据多个列进行排序
# 先按Age升序，然后按Score降序
df_sorted_multiple = df.sort_values(by=['Age', 'Score'], ascending=[True, False])
print("\\n多列排序（先按Age升序，再按Score降序）:")
print(df_sorted_multiple)
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.linalg import norm
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
# TF-IDF算法
def calculate_tfidf_cosine_similarity(text1, text2):
    vectorizer = TfidfVectorizer(max_df=0.85, min_df=2)
    corpus = [text1, text2]
    vectors = vectorizer.fit_transform(corpus)
    similarity = cosine_similarity(vectors)
    return similarity[0][1]

def tfidf_similarity_list(list1, list2):
    def add_space(s):
        if isinstance(s, str):
            return ' '.join(list(s))
        else:
            return ''
    # 合并两个文本列表
    corpus = list1 + list2
    cv = TfidfVectorizer()
    vectors = cv.fit_transform(corpus).toarray()

    # 计算TF系数
    sim_matrix = np.dot(vectors[:len(list1)], vectors[len(list1):].T) / (
        np.outer(norm(vectors[:len(list1)], axis=1), norm(vectors[len(list1):], axis=1)))

    # 添加打印语句
    print("Similarity Matrix:")
    print(sim_matrix)

    return sim_matrix

# 从excel文件读取数据
df2=pd.read_csv('ontologyLabels_5all.tsv', sep='\t')
df1=pd.read_csv('ddtoLabel.tsv', sep='\t')[300:305]
# df2=pd.read_csv('./ontologyLabels_5all.tsv', sep='\t')[0:10000]
# df1=pd.read_csv('./ddtoLabel.tsv', sep='\t')[10:20]
# 提取两列文本数据
column1= df1['label'].tolist()
column2 = df2['label'].tolist()
link2 = df2['link'].tolist()
print(len(set(column2)))
# 存储最终结果
result_text1 = []  # 存储最佳匹配文本
result_text2 = []
result_scores = []  # 存储相似性得分
result_links = []  # 存储匹配到的链接


# 遍历 column1 中的每个单元格
for text1 in tqdm(column1):
    #判断column1中的单元格是否为空，如果为空则终止，
    if not pd.isnull(text1) :
        list1 = eval(text1)
        print(list1)
        # print(column2)
        result = tfidf_similarity_list(list1,column2)

        # 获取最高相似度分数和对应的字符串
        best_score_index = np.unravel_index(np.argmax(result, axis=None), result.shape)
        best_score = result[best_score_index]
        print("Best Match Index:", best_score_index)
        print("Best Match Score:", best_score)
        # best_result = [list1[best_score_index[0]], column2[best_score_index[1]]]
        best_result = [list1, column2[best_score_index[1]]]
        best_link = link2[best_score_index[1]]
        print(best_score_index)
        # 记录每行数据
        result_text1.append(best_result[0])
        result_text2.append(best_result[1])
        result_links.append(best_link)
        result_scores.append(best_score)
# 将result_data作为新列写入excel文件
df3 = pd.DataFrame(data={"source":result_text1,"target":result_text2,"score":result_scores, "link": result_links})
# df['result_text2'] = result_text2
# df['result_score'] = result_scores
df3.to_excel(".xlsx")

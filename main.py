import numpy as np
import pandas as pd

class TestUserMap:

    def __init__(self):
        self.map = {}

    def get_map(self):
        return self.map

    def put_user(self, user_id, test_user):
        self.map[user_id] = test_user

    def get_user(self, user_id):
        return self.map.get(user_id)

class TestUser:

    def __init__(self, user_id):
        self.user_id = user_id
        # the list of rated movie
        self.list_of_rated_movie = []
        # the list of rate of rated movie
        self.list_of_rate_of_rated_movie = []
        # the list of unrated movie
        self.list_of_unrated_movie = []

    def get_user_id(self):
        return self.user_id

    def get_list_of_rated_movie(self):
        return self.list_of_rated_movie

    def get_list_of_rate_of_rated_movie(self):
        return self.list_of_rate_of_rated_movie

    def get_list_of_unrated_movie(self):
        return self.list_of_unrated_movie


def build_test_user_map(test_file):
    '''
    build the test user list from test.txt
    :param test_file: test.txt
    :return: python dictionary
    '''
    df = pd.read_table(test_file, sep="\s+", header=None)

    rows = df.shape[0]

    # create a map to store the user id with TestUser object
    user_map = TestUserMap().get_map()

    for i in range(rows):
        user_id = df[0][i]

        if user_id not in user_map:
            user = TestUser(user_id)
            user_map[user_id] = user

        user = user_map[user_id]
        rate = df[2][i]
        if rate > 0:
            user.get_list_of_rated_movie().append(df[1][i])
            user.get_list_of_rate_of_rated_movie().append(df[2][i])
        else:
            user.get_list_of_unrated_movie().append(df[1][i])

    return user_map

def build_train_matrix(train_file, train_matrix):
    '''
    load the training data
    :param train_file: txt file
    :param matrix: python 2-d array
    :return: void
    '''
    file = open(train_file, "r")
    lines_of_file = file.read().strip().split("\n")

    for i in range(len(lines_of_file)):
        line = lines_of_file[i]
        train_matrix[i] = [int(val) for val in line.split()]


def avg_movie_rate_of_train_users(train_matrix):
    '''
    calculate the mean of each train user in the train data
    :param train_matrix: python 2-d array
    :return: python dictionary, K: train user id, V: mean of given train user
    '''

    map_mean_train_users = {}
    for index, row in enumerate(train_matrix):
        mean_rate = 0.0

        user_id = index + 1
        non_zero_list = [rate for rate in row if rate > 0]

        if len(non_zero_list) > 0:
            mean_rate = sum(non_zero_list) / len(non_zero_list)

        map_mean_train_users[user_id] = mean_rate

    return map_mean_train_users
def avg_movie_rate_of_train_movies(train_matrix):
    '''
    calculate the mean rate of each movie in the train
    :param movie_id: int
    :param train_matrix: python 2-d list
    :return: python dictionary, K: movie id, V: mean rate
    '''

    map_mean_rate = {}

    t_train_matrix = np.array(train_matrix).T

    for index, row in enumerate(t_train_matrix):
        movie_id = index + 1
        mean_rate = 0.0

        non_zero = [rate for rate in row if rate > 0]
        if len(non_zero) > 0:
            mean_rate = sum(non_zero) / len(non_zero)

        map_mean_rate[movie_id] = mean_rate

    return map_mean_rate
def avg_movie_rate_of_test_user(user_id, test_map):
    '''
    calculate the average rate of given test user
    :param user_id: int
    :param test_map: python dictionary
    :return: int
    '''
    user = test_map[user_id]
    list_of_rate_of_rated_movie = user.get_list_of_rate_of_rated_movie()

    avg_rate = 0.0
    if len(list_of_rate_of_rated_movie) != 0:
        avg_rate = sum(list_of_rate_of_rated_movie) / len(list_of_rate_of_rated_movie)

    return avg_rate


def cal_cosine_similarity(v1, v2):
    '''
    calculate the cosine similarity
    :param v1: python list
    :param v2: python list
    :return: int
    '''
    if len(v1) != len(v2) or len(v1) == 0 or len(v2) == 0:
        print("Error: invalid input, the length of vectors should be equal and larger than 0")
        return

    result = 0.0
    numerator = 0.0
    denominator = 0.0

    # numerator = np.dot(v1, v2)
    # denominator = np.sqrt(np.dot(v1, v1)) * np.sqrt(np.dot(v2, v2))

    numerator = np.inner(v1, v2)
    denominator = np.linalg.norm(v1) * np.linalg.norm(v2)

    if denominator != 0.0:
        result = numerator / denominator

    return result

def find_similar_neighbor_cosine(user_id, train_matrix, test_map):
    '''
    find the top k neighbors with cosine similarity
    :param user_id: int
    :param train_matrix: python 2-d array
    :param test_map: python dictionary
    :return: list_of_tuple(user_id, cosine_similarity)
    '''

    test_user = test_map[user_id]
    list_of_unrated_movie = test_user.get_list_of_unrated_movie()
    list_of_rated_movie = test_user.get_list_of_rated_movie()
    list_of_rate_of_rated_movie = test_user.get_list_of_rate_of_rated_movie()

    list_of_neighbor = []

    # find the neighbor
    # go through the 200 users
    for row in range(len(train_matrix)):
        train_user_id = row + 1

        # skip the target user itself
        if train_user_id == user_id: continue

        common_movie = 0

        numerator = 0.0
        denominator = 0.0
        cosine_similarity = 0.0

        test_vector = []
        train_vector = []

        for i in range(len(list_of_rated_movie)):
            movie_id = list_of_rated_movie[i]
            test_movie_rate = list_of_rate_of_rated_movie[i]
            train_movie_rate = train_matrix[train_user_id - 1][movie_id - 1]

            # movie rate with zero means the user doesn't rate the movie
            # we should not consider it as part of the calculation of cosine similarity
            if train_movie_rate != 0:
                common_movie += 1

                test_vector.append(test_movie_rate)
                train_vector.append(train_movie_rate)

        # the common movie between test data and train data should be larger than 1
        if common_movie > 1:
            cosine_similarity = cal_cosine_similarity(test_vector, train_vector)

            list_of_neighbor.append((train_user_id, cosine_similarity))
        if common_movie < 2:
            list_of_neighbor.append([0])

    #list_of_neighbor.sort(key=lambda tup : tup[1], reverse=True)

    return list_of_neighbor

def cal_pearson_correlation(test_users, test_mean_rate, train_users, train_mean_rate):
    '''
    calculate the pearson correlation
    :param test_users: list of tuple(movie id, movie rate) from unrated movie in test data
    :param test_mean_rate: mean of rate in the test user
    :param train_users: python list
    :param train_mean_rate: mean of rate in the train user
    :return: float
    '''
    # there are 3 cases:
    # case 1: result > 0
    #           return result
    # case 2.1: result == 0, because of no common component
    #           return 0
    # case 2.2: result == 0, because of in one or two of vector, each component == mean of vector
    #           return the mean of vector

    numerator = 0.0
    denominator = 0.0

    # filter the common components as vector
    vector_test_rates = []
    vector_train_rates = []

    for test_movie_id, test_movie_rate in test_users:
        train_movie_rate = train_users[test_movie_id - 1]
        if train_movie_rate > 0 and test_movie_rate > 0:
            vector_test_rates.append(test_movie_rate)
            vector_train_rates.append(train_movie_rate)

    # no common component
    if len(vector_train_rates) == 0 or len(vector_test_rates) == 0:
        return 0.0

    adj_vector_test_users = [movie_rate - test_mean_rate for movie_rate in vector_test_rates]
    adj_vector_train_users = [movie_rate - train_mean_rate for movie_rate in vector_train_rates]

    numerator = np.inner(adj_vector_train_users, adj_vector_train_users)
    denominator = np.linalg.norm(adj_vector_test_users) * np.linalg.norm(adj_vector_train_users)

    # each component of one or both vectors is the same
    if denominator == 0.0:
        return 0.0

    return numerator / denominator

def find_similar_neighbor_pearson(user_id, train_matrix, test_map, train_mean_rate_map):
    '''
    calculate all neighbors's pearson correlation for given test user
    :param user_id: int
    :param train_df: python 2-d array
    :param test_map: python dictionary
    :param train_mean_rate_map: python dictionary, the mean of each train user
    :return: a list of tuple(user id, similarity)
    '''
    list_of_neighbors = []

    # average rate of given test user
    avg_movie_rate_in_test = avg_movie_rate_of_test_user(user_id, test_map)

    user = test_map[user_id]
    list_of_rated_movie = user.get_list_of_rated_movie()
    list_of_rate_of_rated_movie = user.get_list_of_rate_of_rated_movie()
    list_of_unrated_movie = user.get_list_of_unrated_movie()

    # zipped_list_of_rated_movie_with_rate = list(zip(list_of_rated_movie, list_of_rate_of_rated_movie))
    zipped_list_of_rated_movie_with_rate = []
    for i in range(len(list_of_rated_movie)):
        zipped_list_of_rated_movie_with_rate.append((list_of_rated_movie[i], list_of_rate_of_rated_movie[i]))

    # find the neighbor
    # go through the 200 users
    for index, row in enumerate(train_matrix):
        train_user_id = index + 1

        # skip the target user
        if train_user_id == user_id: continue

        # average rate of given train user
        avg_movie_rate_in_train = train_mean_rate_map[train_user_id]

        pearson_correlation = cal_pearson_correlation(zipped_list_of_rated_movie_with_rate, avg_movie_rate_in_test, row, avg_movie_rate_in_train)

        # correct the pearson correlation
        # range is [-1, 1]
        if pearson_correlation > 1.0:
            pearson_correlation = 1.0
        if pearson_correlation < -1.0:
            pearson_correlation = -1.0

        if pearson_correlation != 0.0:
            list_of_neighbors.append((train_user_id, pearson_correlation))

        if pearson_correlation == 0.0:
            list_of_neighbors.append([0.0])

    # list_of_neighbors.sort(key=lambda tup: tup[1], reverse=True)

    return list_of_neighbors

def cal_adjusted_cosine_similarity(target_id, neighbor_id, t_train_matrix, train_mean_rate_map):
    '''
    calculate the adjusted cosine similarity in item-based approach
    :param target_id: int, the given movie id
    :param neighbor_id: the certain movie id in the train data
    :param t_train_matrix: the transposed train matrix
    :param train_mean_rate_map: mean rate of each user in the train data
    :return: float
    '''
    adj_cosine_sim = 0.0

    target_row = t_train_matrix[target_id - 1]
    neighbor_row = t_train_matrix[neighbor_id - 1]

    # filter the common component
    # subtract the mean rate
    target_vector = []
    neighbor_vector = []
    for i in range(len(t_train_matrix[0])):
        if target_row[i] > 0 and neighbor_row[i] > 0:
            target_vector.append(target_row[i] - train_mean_rate_map[i + 1])
            neighbor_vector.append((neighbor_row[i]) - train_mean_rate_map[i + 1])

    # calculate the adjusted cosine similarity
    # the common component should be larger than 1
    if len(target_vector) == len(neighbor_vector) and len(target_vector) > 1:
        adj_cosine_sim = cal_cosine_similarity(target_vector, neighbor_vector)

    return adj_cosine_sim

def build_map_similar_neighbor_adj_cosine(train_matrix, train_mean_rate_map):
    '''
    build map of the similar neighbors based on adjusted cosine similarity
    :param movie_id: int
    :param train_matrix: python 2-d array
    :param train_mean_rate_map: python dictionary, K: train user id, V: mean rate of given train user
    :return: python dictionary, K: movie id, V: python dictionary(K: movie id, V: similarity)
    '''
    neighbor_map = {}

    # transpose the train matrix
    t_train_matrix = np.array(train_matrix).T
    print(len(t_train_matrix))
    for i in range(len(t_train_matrix)):
        target_id = i + 1

        if target_id not in neighbor_map:
            neighbor_map[target_id] = {}

        for j in range(i + 1, len(t_train_matrix)):
            neighbor_id = j + 1

            adj_cosine_sim = cal_adjusted_cosine_similarity(target_id, neighbor_id, t_train_matrix, train_mean_rate_map)

            neighbor_map[target_id][neighbor_id] = adj_cosine_sim

            if neighbor_id not in neighbor_map:
                neighbor_map[neighbor_id] = {}

            neighbor_map[neighbor_id][target_id] = adj_cosine_sim

    return neighbor_map


def runtest(testfile,\
        train_matrix,\
        train_mean_rate_map,\
        train_movie_mean_map,\
        adj_cosine_map_of_neighbors):
    '''
    process input file
    :param io_file: python tuple(output file, input file)
    :param train_matrix: python 2-d list
    :param train_mean_rate_map: python dictionary
    :param train_movie_mean_map: python dictionary
    :param iuf_train_matrix: python 2-d list
    :return: void
    '''

    test_map = build_test_user_map(testfile)

    # sort the test users
    list_of_test_user_id = sorted(test_map.keys())

    for user_id in list_of_test_user_id:
        user = test_map[user_id]
        list_of_unrated_movie = user.get_list_of_unrated_movie()

        # neighbor searching based on cosine similarity
        #根據test的userid去計算每個sim 沒有共同movie的sim設為0
        cosine_list_of_neighbors = find_similar_neighbor_cosine(user_id, train_matrix, test_map)
        #print(len(cosine_list_of_neighbors)) 
        
        # neighbor searching based on pearson correlation
        pearson_list_of_neighbors = find_similar_neighbor_pearson(user_id, train_matrix, test_map, train_mean_rate_map)###計算sim單獨一個user
        #print(len(pearson_list_of_neighbors))

def main():
    '''
    the main entry the program
    :return: void
    '''
    
    train_file = "train.txt"
    testfile = "test5.txt"
    # cross-validation
    # io_list = [("./mae/eval_result.txt", "./mae/eval_test.txt")]
    # train_file = "./mae/eval_train.txt"

    # build the train matrix from train.txt
    num_of_users = 200
    num_of_movies = 1000 
    train_matrix = [[0] * num_of_movies] * num_of_users
    build_train_matrix(train_file, train_matrix)

    print(len(train_matrix))
    train_mean_rate_map = avg_movie_rate_of_train_users(train_matrix)
    print(len(train_mean_rate_map))
    train_movie_mean_map = avg_movie_rate_of_train_movies(train_matrix)
    print(len(train_movie_mean_map))

    # build neighbor map based on adjusted cosine similarity
    print(len(train_matrix), len(train_mean_rate_map))
    adj_cosine_map_of_neighbors = build_map_similar_neighbor_adj_cosine(train_matrix, train_mean_rate_map)
    #1000*1000 itembase-------------------------------------------------------------------------------------------
    print("finish building the neighbor map")
    print(len(adj_cosine_map_of_neighbors))
    #print(adj_cosine_map_of_neighbors)

    print("-----------------------------------------------------")

    
    runtest(testfile,\
        train_matrix,\
        train_mean_rate_map,\
        train_movie_mean_map,\
        adj_cosine_map_of_neighbors)

main()
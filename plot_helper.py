import matplotlib.pyplot as plt

def draw_boxplot_comparision(model_name,score_array):
    fig = plt.figure()
    fig.suptitle("Boxplot model comparison:")
    ax = fig.add_subplot(111)
    ax.set_title(model_name)
    ax.boxplot(score_array)
    plt.show()

def draw_linear_comparision(model_name,features,actual_values, predicted_values):
    fig = plt.figure()
    fig.suptitle("R2 model comparison")
    ax = fig.add_subplot(111)
    ax.set_title(model_name)
    ax.plot(actual_values, predicted_values)
    ax.scatter(features, actual_values)
    plt.show()

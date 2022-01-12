import matplotlib.pyplot as plt

def draw_boxplot_comparision(models,score_array):
    fig = plt.figure()
    fig.suptitle("Boxplot model comparison:")
    ax = fig.add_subplot()
    ax.boxplot(score_array)
    model_names = []
    for model in models:
        model_names.append(model[0])
    ax.set_xticklabels(model_names, rotation=45)
    plt.show()

def draw_linear_comparision(model_name,features,actual_values, predicted_values):
    fig = plt.figure()
    fig.suptitle("R2 model comparison")
    ax = fig.add_subplot()
    ax.set_title(model_name)
    ax.plot(actual_values, predicted_values)
    ax.scatter(features, actual_values)
    plt.show()

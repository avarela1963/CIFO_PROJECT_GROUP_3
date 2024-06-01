import pandas as pd
import matplotlib.pyplot as plt

def plot_fitness_results():
    files = [
        ('fitness_results_10000_fac.csv', 10000, 10),  # Add actual success ratio here
        ('fitness_results_10000_med.csv', 10000, 0),  # Add actual success ratio here
        ('fitness_results_10000_dif.csv', 10000, 6.7)  # Add actual success ratio here
    ]

    plt.figure()

    colors = ['blue', 'orange', 'green']  # Assuming the third plot is green

    for index, (csv_file, population_size, success_ratio) in enumerate(files):
        # Load the DataFrame from the CSV file
        df = pd.read_csv(csv_file)

        # Extract the last row which contains the average fitness values
        average_row = df.iloc[-1]

        # Adjust the x-axis values from 1 to the length of the average_row
        x_values = range(1, len(average_row) + 1)

        # Get the last fitness value
        last_fitness_value = average_row.iloc[-1]

        # Plot the average data
        plt.plot(x_values, average_row.values, marker='o', linestyle='-', linewidth=2,
                 label=f'File: {csv_file} - Size of population = {population_size}', color=colors[index])

        # Adjust font size for text boxes
        font_size = 8

        if colors[index] == 'blue':
            # Add text box further above the curve for the blue plot
            plt.text(x_values[-1], average_row.values[-1] + 1.0, f'Final average fitness = {last_fitness_value:.2f}',
                     fontsize=font_size, verticalalignment='bottom', horizontalalignment='right',
                     bbox=dict(facecolor='white', alpha=0.5))

            plt.text(x_values[-1], average_row.values[-1] + 0.1, f'Success rate = {success_ratio}%',
                     fontsize=font_size, verticalalignment='bottom', horizontalalignment='right',
                     bbox=dict(facecolor='white', alpha=0.5))
        elif colors[index] == 'orange':
            # Add text box above the curve for the orange plot
            plt.text(x_values[-1], average_row.values[-1] + 0.4, f'Final average fitness = {last_fitness_value:.2f}',
                     fontsize=font_size, verticalalignment='bottom', horizontalalignment='right',
                     bbox=dict(facecolor='white', alpha=0.5))

            plt.text(x_values[-1], average_row.values[-1] - 0.3, f'Success rate = {success_ratio}%',
                     fontsize=font_size, verticalalignment='top', horizontalalignment='right',
                     bbox=dict(facecolor='white', alpha=0.5))
        else:
            # Add text box below the curve for other plots
            plt.text(x_values[-1], average_row.values[-1] - 0.5, f'Final average fitness = {last_fitness_value:.2f}',
                     fontsize=font_size, verticalalignment='top', horizontalalignment='right',
                     bbox=dict(facecolor='white', alpha=0.5))

            plt.text(x_values[-1], average_row.values[-1] - 1.4, f'Success rate = {success_ratio}%',
                     fontsize=font_size, verticalalignment='top', horizontalalignment='right',
                     bbox=dict(facecolor='white', alpha=0.5))

    # Add the text box to the top left corner of the plot (uncomment if needed)
    # plt.text(0.03, 0.9, 'runs with tournament selection',
    #          fontsize=12, verticalalignment='top', horizontalalignment='left',
    #          transform=plt.gca().transAxes,
    #          bbox=dict(facecolor='white', alpha=0.5))

    # Display the plot with the legend
    plt.xlabel('Iteration')  # Label x-axis as 'Iteration'
    plt.ylabel('Average Fitness')
    plt.title('Average Fitness over 30 runs')
    plt.legend(loc='upper left')  # Add the legend to the plot in the top left corner
    plt.grid(True)  # Optionally add a grid
    plt.show()  # Ensure the plot is shown

if __name__ == "__main__":
    # Plot the fitness results
    plot_fitness_results()
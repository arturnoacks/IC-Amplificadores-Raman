import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
from nsga2 import NSGA2, fast_non_dominated_sort
from analytic_solver import evaluate_analytic_amp
import csv

def main():
    n_pumps = 3

    power_max_values = [1.0, 1.5, 2.0, 2.5]
    
    fiber_lengths = [2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0, 22.5, 25.0, 27.5, 30.0]
    
    colors = ['r', 'b', 'm', 'g'] 
    
    gain_results = {}
    ripple_results = {}

    n_generations = 150

    for p_max in power_max_values:
        gains_current_curve = []
        ripple_current_curve = []
        print(f"\n--- Simulando curva para Pmax = {p_max}W ---")
        
        for length in fiber_lengths:
            nsga2 = NSGA2(
                pop_size=200, 
                n_pumps=n_pumps, 
                mutation_rate=0.3, 
                power_max=p_max,
                fiber_len=length
            )
            
            population = nsga2.initialize_population()
        
            population, history_best_gain, history_best_ripple = nsga2.evolve(population, evaluate_analytic_amp, n_generations=n_generations)

            objectives = np.array([
                nsga2.evaluate_objectives(ind, evaluate_analytic_amp)
                for ind in population
            ])

            # salva pareto
            fronts = fast_non_dominated_sort(population, objectives)
            pareto_front = fronts[0]
            pareto_ripple = [-objectives[i][0] for i in pareto_front]
            pareto_gain = [objectives[i][1] for i in pareto_front]
            
            plt.figure(figsize=(6, 4))

            plt.scatter(pareto_ripple, pareto_gain, s=15)

            sorted_pairs = sorted(zip(pareto_ripple, pareto_gain))
            r_sorted, g_sorted = zip(*sorted_pairs)
            plt.plot(r_sorted, g_sorted, linewidth=1)

            plt.xlabel('Ripple (dB)')
            plt.ylabel('Gain (dB)')
            plt.title(f'Pareto Front - Pmax={p_max}W, L={length}m')

            plt.grid(True, linestyle='--', alpha=0.4)
            plt.tight_layout()

            plt.savefig(f'pareto_pmax_{p_max}_len_{length:.1f}.png', dpi=300)
            plt.close()

            csv_pareto = f"pareto_pmax_{p_max}_len_{length:.1f}.csv"

            with open(csv_pareto, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['gain_dB', 'ripple_dB'])

                for g, r in zip(pareto_gain, pareto_ripple):
                    writer.writerow([g, r])

            # salva o resto igual ga
            best_gain = history_best_gain[len(history_best_gain)-1]
            best_ripple = history_best_ripple[len(history_best_gain)-1]
            
            gains_current_curve.append(best_gain)
            ripple_current_curve.append(best_ripple)
            print(f"L={length}m -> Ganho: {best_gain:.2f} dB, Ripple: {best_ripple:.2f}")

            # Salva gráfico de evolução
            plt.figure(figsize=(6, 4))
            plt.plot(range(1, len(history_best_gain) + 1), history_best_gain, markersize=3, linewidth=1)
            plt.xlabel('Geração', fontsize=10)
            plt.ylabel('Melhor ganho (dB)', fontsize=10)
            plt.title(f'Pmax={p_max}W - L={length}m', fontsize=11)
            plt.grid(True, linestyle='--', alpha=0.4)
            plt.tight_layout()
            plot_filename = f"evolution_pmax_{p_max}_len_{length:.1f}.png"
            plt.savefig(plot_filename, dpi=300)
            plt.close()

            csv_filename = f"evolution_pmax_{p_max}_len_{length:.1f}.csv"
            with open(csv_filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['generation', 'best_gain_dB'])
                for gen, gain in enumerate(history_best_gain, start=1):
                    writer.writerow([gen, gain])
                writer.writerow(['best_individual [lambda, power]'])
                writer.writerow([history_best_gain])
                writer.writerow(['ripple'])
                writer.writerow([best_ripple])


        gain_results[p_max] = gains_current_curve
        ripple_results[p_max] = ripple_current_curve

        
        filename = f'gain_data_pmax_{p_max}.csv'
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Fiber Length (m)', 'Average Gain (dB)', 'Average Ripple (dB)'])
            for length, gain, ripple in zip(fiber_lengths, gains_current_curve, ripple_current_curve):
                writer.writerow([length, gain, ripple])



    # grafico de ganhos
    plt.figure(figsize=(7, 5))
    
    for i, p_max in enumerate(power_max_values):
        y_values = gain_results[p_max]
        plt.plot(fiber_lengths, y_values, 
                 marker='*',
                 markersize=6,
                 color=colors[i],
                 linewidth=1, 
                 label=f'Pmax = {p_max}W')

    plt.xlabel('Tellurite fiber length [m]', fontsize=12)
    plt.ylabel('Average Gain [dB]', fontsize=12)
    
    plt.legend(loc='upper left', frameon=True, edgecolor='gray', fancybox=False)
    
    plt.xlim(0, 31)
    plt.ylim(0, 15)
    
    plt.tick_params(direction='in', top=True, right=True)
    
    plt.tight_layout()
    plt.savefig(f'average_gain_vs_fiber_length_{n_pumps}_pumps.png', dpi=300)


    # grafico de ripple
    plt.figure(figsize=(7, 5))
    
    for i, p_max in enumerate(power_max_values):
        y_values = ripple_results[p_max]
        plt.plot(fiber_lengths, y_values, 
                 marker='s',
                 markersize=6,
                 color=colors[i],
                 linewidth=1, 
                 label=f'Pmax = {p_max}W')

    plt.xlabel('Tellurite fiber length [m]', fontsize=12)
    plt.ylabel('Average ripple [dB]', fontsize=12)
    
    plt.legend(loc='upper left', frameon=True, edgecolor='gray', fancybox=False)
    
    plt.xlim(0, 31)
    plt.ylim(0, 5)
    
    plt.tick_params(direction='in', top=True, right=True)
    
    plt.tight_layout()
    plt.savefig(f'average_ripple_vs_fiber_length_{n_pumps}_pumps.png', dpi=300)


if __name__ == "__main__":
    main()
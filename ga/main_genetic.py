import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
from genetic_algorithm import GeneticAlgorithm
from analytic_solver import evaluate_analytic_amp
from numeric_solver import evaluate_bvp_amp
import time
import csv
import os

def main():

    start_time = time.perf_counter()

    n_pumps = 3

    power_max_values = [1.0, 1.5, 2.0, 2.5]
    
    fiber_lengths = [2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0, 22.5, 25.0, 27.5, 30.0]
    
    colors = ['r', 'b', 'm', 'g'] 
    
    gain_results = {}
    ripple_results = {}

    n_generations = 3000

    os.makedirs("data", exist_ok=True)

    for p_max in power_max_values:
        gains_current_curve = []
        ripple_current_curve = []
        print(f"\n--- Simulando curva para Pmax = {p_max}W ---")
        
        for length in fiber_lengths:
            ga = GeneticAlgorithm(
                pop_size=50, 
                n_pumps=n_pumps, 
                mutation_rate=0.3, 
                power_max=p_max,
                fiber_len=length
            )
            
            population = ga.initialize_population()
        
            population, anl_fitness_scores, num_fitness_scores, best_individual, best_gain, best_ripple, best_history = ga.evolve(population, evaluate_analytic_amp, evaluate_bvp_amp, n_generations=n_generations)
            
            gains_current_curve.append(best_gain)
            ripple_current_curve.append(best_ripple)
            print(f"L={length}m -> Ganho: {best_gain:.2f} dB, Ripple: {best_ripple:.2f}")

            # Salva gráfico de evolução
            plt.figure(figsize=(6, 4))
            plt.plot(range(1, len(best_history) + 1), best_history, markersize=3, linewidth=1)
            plt.xlabel('Geração', fontsize=10)
            plt.ylabel('Melhor ganho (dB)', fontsize=10)
            plt.title(f'Pmax={p_max}W - L={length}m', fontsize=11)
            plt.grid(True, linestyle='--', alpha=0.4)
            plt.tight_layout()
            plot_filename = f"data/evolution_pmax_{p_max}_len_{length:.1f}.pdf"
            plt.savefig(plot_filename, dpi=300)
            plt.close()

            # csv do melhor individuo + população
            csv_filename = f"data/evolution_pmax_{p_max}_len_{length:.1f}.csv"
            with open(csv_filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['--- BEST INDIVIDUAL ---'])
                writer.writerow(['generation', 'anl_fitness_score'])
                for gen, gain in enumerate(best_history, start=1):
                    writer.writerow([gen, gain])
                writer.writerow(['best_individual [lambda, power]'])
                writer.writerow([best_individual])
                writer.writerow(['ripple'])
                writer.writerow([best_ripple])
            
            with open(csv_filename, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([])
                writer.writerow(['--- LAST GENERATION POPULATION ---'])
                
                # Cria o cabeçalho dos indivíduos
                num_params = len(population[0])
                headers = ['individual_id'] + [f'param_{i}' for i in range(num_params)] + ['anl_fitness_score'] + ['num_fitness_score']
                writer.writerow(headers)
                
                # Grava a população inteira
                for idx, (individual, anl_score, num_score) in enumerate(zip(population, anl_fitness_scores, num_fitness_scores)):
                    row = [idx] + list(individual) + [anl_score] + [num_score]
                    writer.writerow(row)


        


        gain_results[p_max] = gains_current_curve
        ripple_results[p_max] = ripple_current_curve

        
        filename = f'data/gain_data_pmax_{p_max}.csv'
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
    plt.savefig(f'data/average_gain_vs_fiber_length_{n_pumps}_pumps.pdf', dpi=300)


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
    plt.savefig(f'data/average_ripple_vs_fiber_length_{n_pumps}_pumps.pdf', dpi=300)


    end_time = time.perf_counter()

    with open("data/time.txt", "w", encoding="utf-8") as file:
        file.write(f"Tempo de execução: {end_time - start_time} segundos")


if __name__ == "__main__":
    main()
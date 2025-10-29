import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid

lambdas = np.linspace(1520,1600,20)
num = 100 # num de testes
ripple_bom = 3 # dB
comprimentosdeonda = np.arange(1360, 1451, 5)
n_pumps = 3

# arrays para salvar os resultados
ripple_arr = np.zeros(num)
ganho_medio_arr = np.zeros(num)
ganho_on_off_medio_arr = np.zeros(num)

varlamb = 5 # o lambda ira variar entre (lambdap + - varlamb/2)
Plp = np.ones(n_pumps) * 0.3 # valor inicial de potencia

varp = 2.2 # o valor de potencia ira variar entre (plp + - p_var/2)

Plpinicial = Plp.copy()

Bws = 0.2 # (nm) largura de banda do analizador de espectro óptico
Fpl = 2 # fator de polarização do laser

distSMF = 10 # comprimento da fibra?
# -------------------------------------------------------------- #

ns = len(lambdas) # numero de sinais
nc = len(comprimentosdeonda) # possiveis lambda

# variação dos dados de entrada
lambaleatorio = np.zeros((num, n_pumps))
potaleatorio = np.zeros((num, n_pumps))
lamb_arr = np.zeros((num, n_pumps))
pot_arr = np.zeros((num, n_pumps))

for i in range(num):
    # Variando os dados de entrada
    # Escolhe np índices aleatórios de comprimentosdeonda
    indices_escolhidos = np.random.randint(0, nc, n_pumps)
    lambdap = comprimentosdeonda[indices_escolhidos]

    lambaleatorio = varlamb * (np.random.rand(n_pumps) - 0.5)
    lambdap = lambdap + lambaleatorio
    potaleatorio[i, :] = np.random.rand(n_pumps) * varp
    Plp = Plpinicial + potaleatorio[i, :]
    lambdap = np.sort(lambdap)
    lamb_arr[i, :] = lambdap
    pot_arr[i, :] = Plp

    fp = (2.99792458e8/lambdap)*10**9  #Hz
    fs = (2.99792458e8/lambdas)*10**9  #Hz
    wp= 2*np.pi*fp
    ws= 2*np.pi*fs

    # Dados da fibra de telúrio

    lamb1 = np.array([1350.6, 1352.5, 1356.9, 1359.6, 1363.9, 1368.5, 1373.8, 1375.7, 1377.6, 1380.9, 1384.2, 1385.9, 1388.5])
    lamb2 = np.array([1390.8, 1392.3, 1395.8, 1397.8, 1400.6, 1402.4, 1405.5, 1407.8, 1411.4, 1413.5, 1415.7, 1419.4, 1421.6])
    lamb3 = np.array([1422.8, 1426.3, 1428.3, 1430.4, 1433.0, 1434.3, 1436.8, 1438.7, 1442.3, 1444.9, 1448.3, 1450.6, 1455.0])
    lamb4 = np.array([1457.3, 1459.3, 1461.0, 1462.3, 1465.0, 1466.6, 1470.1, 1471.7, 1473.4, 1475.7, 1477.5, 1479.4, 1480.7])
    lamb5 = np.array([1482.6, 1484.5, 1486.8, 1488.4, 1490.4, 1492.5, 1494.4, 1496.0, 1497.2, 1499.1, 1501.4, 1503.0, 1504.4])
    lamb6 = np.array([1506.7, 1508.7, 1510.5, 1512.5, 1514.5, 1516.6, 1518.3, 1519.7, 1522.0, 1524.4, 1526.3, 1528.6, 1530.4])
    lamb7 = np.array([1531.8, 1533.6, 1535.9, 1537.3, 1538.9, 1540.5, 1542.3, 1543.1, 1545.0, 1547.2, 1549.5, 1550.9, 1552.9])
    lamb8 = np.array([1554.4, 1556.4, 1558.7, 1560.6, 1562.5, 1564.2, 1565.5, 1567.1, 1568.5, 1570.3, 1572.3, 1574.8, 1577.1])
    lamb9 = np.array([1579.2, 1581.6, 1583.5, 1585.7, 1588.1, 1590.3, 1592.7, 1595.0, 1597.3, 1599.9, 1602.8, 1604.6, 1606.9])
    lamb10= np.array([1610.0, 1611.5, 1613.9, 1616.4, 1618.5, 1620.6, 1623.3, 1625.7, 1629.4, 1633.0, 1640.3, 1645.7, 1650.0])
    # comprimentos de onda em nm
    lamb_data = np.concatenate([lamb1, lamb2, lamb3, lamb4, lamb5, lamb6, lamb7, lamb8, lamb9, lamb10])

    alfa1 = np.array([29.5543, 29.1659, 29.3709, 29.0644, 28.5740, 28.5541, 28.8409, 29.0660, 29.4548, 29.5983, 29.5782, 29.1489, 28.5970])
    alfa2 = np.array([28.2700, 27.9430, 27.2890, 27.0233, 26.9213, 26.6966, 26.5128, 26.3086, 26.1045, 25.9002, 25.6754, 25.3691, 25.1853])
    alfa3 = np.array([24.9604, 24.8585, 24.8996, 24.6545, 24.4502, 24.4708, 24.4711, 24.2872, 24.3080, 24.3083, 24.3086, 24.4929, 24.5752])
    alfa4 = np.array([24.5959, 24.7392, 24.8825, 24.9236, 24.9443, 25.0058, 25.3743, 25.3540, 25.2928, 25.2726, 25.2523, 25.1707, 25.1095])
    alfa5 = np.array([25.0074, 24.9667, 24.8852, 24.8853, 24.8037, 24.8039, 24.7387, 24.6652, 24.6163, 24.4447, 24.4695, 24.3470, 24.1999])
    alfa6 = np.array([24.1755, 24.2248, 24.1718, 23.8858, 23.7633, 23.7430, 23.6818, 23.7024, 23.6004, 23.4984, 23.4782, 23.4784, 23.4990])
    alfa7 = np.array([23.3560, 23.3358, 23.2337, 23.1725, 23.1318, 23.0297, 22.9890, 22.9686, 22.9688, 22.9486, 22.9488, 22.9285, 22.8878])
    alfa8 = np.array([22.9084, 22.9700, 22.7657, 22.8273, 22.9502, 22.9503, 23.0936, 23.2165, 22.8485, 22.9714, 23.0534, 23.0332, 23.1971])
    alfa9 = np.array([23.1359, 23.1157, 23.3409, 23.3615, 23.4231, 23.4438, 23.2396, 23.3829, 23.5672, 23.8333, 23.8132, 24.1406, 23.9568])
    alfa10= np.array([24.4479, 24.5707, 24.6323, 24.9393, 24.9804, 25.3078, 25.2672, 25.7992, 26.2903, 26.4134, 27.3549, 28.0671, 26.5951])
    # atenuação em dB/km
    alfa_data = np.concatenate([alfa1, alfa2, alfa3, alfa4, alfa5, alfa6, alfa7, alfa8, alfa9, alfa10]) + 5 - 0.36

    alfasdBkmSMF = np.interp(lambdas, lamb_data, alfa_data)  # interpolação dos dados
    alfapdBkmSMF = np.interp(lambdap, lamb_data, alfa_data)

    alfasSMF = alfasdBkmSMF * np.log(10) * 1e-4 # (neper/m)
    alfapSMF = alfapdBkmSMF * np.log(10) * 1e-4

    alfa_bombeios = np.mean(alfapSMF)  # (neper/m)

    # Calculo do ganho Raman
    lambdap_ref = 1460 # bombeio de referencia utilizado para obter o cr de pico
    Crpico_refp = 1 # 1/(W*km) coeficiente de raman de pico
    Crpico_up_refp = Crpico_refp * 1e-3 # 1/(W*m) unidade padrão

    Crpicopump = lambdap_ref * Crpico_up_refp / lambdap

    # Curva de ganho normalizada
    sepfreq = np.array([0.0218, 0.1102, 0.1542, 0.2204, 0.2866, 0.3526, 0.4321, 0.4849, 0.5644, 0.6305, 0.6833, 0.7627, 0.8155, 0.8816, 0.9477, 1.0138, 1.0800, 1.1461, 1.2388, 1.3183, 1.3577, 1.4372, 1.5170, 1.5967, 1.6897, 1.7827, 1.9024, 2.0220, 2.1150, 2.1813, 2.3143, 2.4073, 2.4735, 2.5931, 2.6595, 2.7125, 2.7789, 2.8853, 2.9517, 3.0447, 3.1112, 3.2442, 3.3773, 3.4304, 3.5369, 3.7766, 3.9365, 4.1097, 4.2430, 4.3363, 4.5096, 4.7094, 4.8293, 4.9759, 5.0692, 5.2424, 5.3224, 5.4290, 5.5356, 5.6688, 5.8153, 5.9884, 6.1083, 6.2149, 6.4414, 6.5481, 6.6814, 6.8180, 6.9512, 7.0845, 7.1845, 7.2679, 7.4012, 7.5678, 7.7010, 7.8509, 7.9842, 8.1174, 8.2005, 8.3669, 8.4834, 8.5999, 8.7997, 8.9328, 9.0493, 9.1658, 9.2489, 9.3487, 9.4485, 9.5483, 9.6480, 9.7312, 9.7643, 9.8641, 9.9639, 10.0303, 10.1134, 10.2632, 10.4463, 10.4962, 10.6460, 10.7791, 10.8789, 10.9619, 11.0449, 11.1280, 11.2111, 11.2941, 11.3439, 11.4270, 11.4934, 11.5264, 11.6928, 11.7425, 11.7756, 11.8753, 11.9250, 12.0413, 12.1411, 12.2241, 12.3406, 12.5236, 12.7567, 12.9399, 13.0398, 13.1565, 13.3065, 13.4565, 13.5733, 13.6735, 13.7738, 13.8074, 13.8743, 13.9579, 14.0248, 14.0752, 14.1255, 14.1925, 14.3260, 14.4098, 14.4269, 14.5107, 14.5776, 14.6612, 14.7283, 14.8453, 14.8790, 14.9628, 15.0466, 15.1136, 15.2307, 15.2812, 15.3484, 15.4989, 15.6159, 15.6665, 15.7837, 15.8173, 15.9509, 16.0513, 16.1515, 16.2350, 16.3018, 16.4351, 16.5181, 16.6177, 16.6673, 16.7670, 16.8667, 16.8995, 17.0158, 17.0821, 17.0651, 17.1647, 17.1809, 17.1971, 17.2967, 17.3629, 17.4291, 17.4787, 17.5284, 17.5445, 17.6442, 17.6439, 17.6602, 17.7432, 17.7595, 17.8424, 17.8419, 17.8916, 18.0079, 18.0409, 18.0572, 18.1068, 18.1732, 18.2062, 18.2057, 18.2386, 18.2882, 18.3044, 18.3540, 18.4202, 18.4697, 18.5856, 18.6352, 18.6514, 18.7676, 18.8171, 18.8834, 18.9329, 19.0325, 19.0654, 19.1150, 19.1979, 19.2973, 19.3800, 19.4795, 19.6958, 19.7121, 19.9452, 20.2450, 20.4447, 20.6944, 20.8274, 20.9271, 21.0433, 21.2095, 21.2590, 21.4251, 21.5581, 21.8077, 21.9574, 22.0071, 22.1736, 22.3569, 22.4569, 22.5905, 22.6906, 22.7409, 22.8578, 22.9584, 23.0256, 23.1593, 23.2100, 23.3275, 23.3780, 23.4618, 23.5122, 23.6126, 23.6464, 23.7140, 23.7816, 23.8488, 23.9161, 23.9833, 24.0508, 24.0849, 24.1522, 24.1862, 24.2534, 24.3376, 24.3885, 24.4559, 24.5232, 24.5906, 24.6747, 24.7590, 24.8267, 24.8607, 24.9446, 24.9954, 25.0795, 25.1637, 25.2645, 25.3486, 25.3659, 25.4496, 25.5001, 25.6171, 25.6676, 25.8015, 25.9019, 26.0521, 26.1691, 26.2861, 26.3864, 26.4865, 26.6200, 26.7701, 26.9034, 27.1369, 27.3535, 27.7033, 27.9366, 28.2698, 28.6196, 29.0694, 29.4026, 29.7358, 30.1022, 30.4520, 30.7685, 31.1183, 31.3349, 31.6347, 31.9346, 32.3010, 32.5509, 32.8674, 33.1672, 33.4337, 33.6003, 33.8168, 34.0834, 34.2499, 34.3998, 34.5830, 34.6997, 34.8163, 34.8995, 34.9495])*10**12
    Crnormal = np.array([0.5836, 1.3020, 2.0204, 2.6939, 3.3404, 4.3102, 5.0645, 5.8726, 6.6539, 7.4351, 8.2701, 9.0514, 9.8595, 10.7215, 11.5297, 12.3379, 13.1191, 13.9542, 14.8163, 15.5975, 16.3787, 17.0791, 17.4564, 17.8337, 18.2379, 18.7229, 19.0195, 19.5315, 19.9358, 20.4207, 20.8250, 21.3101, 21.9297, 22.4417, 22.8190, 23.3039, 23.6811, 23.9777, 24.3010, 24.7591, 25.0556, 25.3522, 25.5949, 25.8913, 26.1609, 26.3500, 26.3504, 26.3778, 26.3782, 26.2976, 26.1903, 26.1908, 26.2450, 26.2454, 26.2456, 26.2461, 26.1924, 26.1927, 26.1930, 26.2472, 26.3553, 26.4904, 26.5446, 26.5987, 26.5993, 26.5187, 26.4114, 26.4117, 26.4120, 26.3114, 26.2780, 26.0762, 26.0429, 25.9423, 25.9426, 25.9767, 25.9770, 26.0784, 26.3479, 26.5167, 26.6180, 26.7866, 26.9891, 27.1915, 27.3264, 27.5287, 27.8319, 28.0005, 28.2028, 28.5060, 28.7419, 28.9441, 29.2136, 29.4831, 29.7191, 30.0222, 30.3254, 30.4941, 30.6629, 30.8651, 30.9328, 31.2361, 31.5057, 31.8762, 32.3141, 32.5837, 32.9205, 33.2911, 33.6279, 33.9647, 34.3352, 34.7056, 35.0764, 35.4469, 35.7836, 36.1878, 36.5920, 36.9963, 37.2995, 37.6364, 37.9060, 38.2768, 38.3447, 38.4125, 38.4128, 38.2111, 38.1105, 38.0099, 37.5725, 37.1351, 36.5967, 36.1255, 35.6543, 35.2505, 34.7121, 34.1399, 33.6350, 32.9955, 32.5919, 31.7168, 31.0098, 30.2357, 29.8319, 29.2598, 28.5193, 27.9136, 27.1730, 26.4326, 25.5912, 24.9517, 24.1103, 23.3361, 22.3937, 21.5187, 20.7784, 19.7349, 18.8935, 18.3549, 17.7829, 17.1099, 16.7398, 16.2687, 16.0669, 15.9999, 16.4714, 16.9767, 17.5828, 17.9197, 18.3240, 19.0647, 19.5363, 20.0415, 20.6474, 21.1190, 21.8597, 22.5331, 23.1393, 23.8128, 24.5200, 25.0587, 25.5302, 26.3382, 26.8435, 27.2475, 27.7862, 28.2914, 28.8974, 29.4699, 30.1769, 30.6820, 31.1200, 31.6251, 32.1301, 32.7026, 33.2077, 33.6791, 34.4198, 35.1269, 35.6993, 36.3727, 36.9788, 37.6523, 38.4604, 39.5043, 40.1441, 40.8175, 41.4238, 42.1309, 42.6360, 43.3768, 44.0504, 44.5891, 45.1953, 45.8688, 46.7444, 47.5862, 48.2598, 48.6307, 49.2367, 49.3720, 49.4738, 49.6426, 49.9462, 50.3169, 50.7212, 51.2938, 51.8665, 52.6746, 53.3484, 53.8201, 54.1574, 54.5281, 54.8649, 54.9999, 54.8657, 54.6977, 54.2604, 53.8903, 53.3854, 52.8471, 51.9047, 50.9622, 50.2556, 48.9764, 47.6637, 46.7548, 45.9471, 45.2065, 44.4325, 43.6919, 42.1098, 40.5613, 39.6525, 38.6427, 37.7675, 36.3537, 35.0409, 33.9637, 32.8865, 31.9777, 30.4629, 29.0154, 27.7699, 26.6928, 25.5483, 24.1682, 22.5861, 20.8693, 19.7584, 18.7487, 17.4022, 16.1904, 14.6756, 13.2956, 12.0501, 11.0065, 10.2661, 9.4582, 8.7179, 7.8764, 6.9677, 6.2273, 5.6554, 5.0497, 4.4104, 3.8720, 3.5692, 3.1992, 2.9303, 2.7286, 2.2916, 2.1911, 2.0574, 1.8896, 1.8568, 1.7567, 1.6569, 1.5567, 1.4566, 1.3902, 1.3574, 1.3582, 1.3255, 1.2587, 1.1921, 1.1256, 1.0928, 0.9925, 1.0270, 0.9941, 1.0284, 0.9952, 0.9621, 0.8954, 0.8958, 0.8962, 0.8967, 0.8297, 0.8300, 0.8302, 0.8303]) 


    # Crnovo entre pumps e sinais
    deltafps = np.abs(fp[:, np.newaxis] - fs)
    Crnovops = np.interp(deltafps, sepfreq, Crnormal)
    Cr_uppsSMF = Crpicopump[:, np.newaxis] * Crnovops # (m/W)

    # Crnovo entre pumps
    deltafpp = np.abs(fp[:, np.newaxis] - fp)
    Crnovopp = np.interp(deltafpp, sepfreq, Crnormal)
    Cr_upppSMF = Crpicopump[:, np.newaxis] * Crnovopp # (m/W)

    # Equação dos pumps com interação entre eles
    z = np.linspace(0, distSMF, 100)  # (m)

    # Propagação dos bombeios
    Pp = np.zeros((n_pumps, len(z)))

    for i_pump in range(n_pumps):
        Pp_p = 0
        Pg_p = 0
        for j_pump in range(n_pumps):
            # frequencia maior
            if j_pump > i_pump:
                Pp_lp = 0
                Pp_gp = 0
            
                for k_pump in range(n_pumps):
                    if k_pump > j_pump:
                        Pp_lp += (wp[j_pump]/wp[k_pump] * Cr_upppSMF[j_pump, k_pump] * Plp[k_pump])
                    
                    if k_pump < j_pump:
                        Pp_gp += Cr_upppSMF[j_pump, k_pump] * Plp[k_pump]

                temp = (-Pp_gp + Pp_lp)
                # proteger divisão por zero (ou por valores muito pequenos)
                tol = 1e-12
                if np.all(np.abs(temp) > tol):
                    # construir o argumento do exp de forma segura e limitar para evitar overflow
                    inner = -alfa_bombeios * (distSMF - z)
                    # inner_exp = 1 - np.exp(inner) * (temp)  <-- original intenção: (1 - exp(inner))*(temp)
                    inner_exp = (1 - np.exp(inner)) * (temp)
                    expo_arg = -1.0 / (alfa_bombeios * Fpl) * inner_exp
                    # limitar expo_arg para um intervalo seguro
                    expo_arg_clipped = np.clip(expo_arg, -700, 700)
                    term = 1 - np.exp(expo_arg_clipped)
                    Pp_p -= (wp[i_pump] / wp[j_pump]) * Cr_upppSMF[i_pump, j_pump] * Plp[j_pump] * term / (temp)
    

            # frequencia menor
            if j_pump < i_pump:
                Np_lp = 0
                Np_gp = 0
                for k_pump in range(n_pumps):
                    if k_pump > j_pump:
                        Np_lp += (wp[j_pump]/wp[k_pump] * Cr_upppSMF[j_pump, k_pump] * Plp[k_pump])            
                    if k_pump < j_pump:
                        Np_gp += Cr_upppSMF[j_pump, k_pump] * Plp[k_pump]
                
                temp = (-Np_gp + Np_lp)
                tol = 1e-12
                if np.all(np.abs(temp) > tol):
                    inner = -alfa_bombeios * (distSMF - z)
                    inner_exp = (1 - np.exp(inner)) * (temp)
                    expo_arg = -1.0 / (alfa_bombeios * Fpl) * inner_exp
                    expo_arg_clipped = np.clip(expo_arg, -700, 700)
                    term = 1 - np.exp(expo_arg_clipped)
                    Pg_p += (Cr_upppSMF[i_pump, j_pump] * Plp[j_pump] * term / (temp))

        
        Pp[i_pump, :] = Plp[i_pump] * np.exp(-alfa_bombeios * (distSMF - z)) * np.exp(Pp_p + Pg_p)


    # Ganho analitico
    Ganho_pump = np.zeros(ns)
    for j_sinal in range(ns):
        for i_pump in range(n_pumps):
            k_val = Cr_uppsSMF[i_pump, j_sinal] / Fpl
            Ganho_pump[j_sinal] += k_val * trapezoid(Pp[i_pump, :], z)

    # Evitar overflow em exp() limitando o argumento
    expo_arg_base = -alfasSMF * distSMF + Ganho_pump
    expo_arg_clipped = np.clip(expo_arg_base, -700, 700)
    Ganho_sem_Raman = np.exp(-alfasSMF * distSMF)
    # proteger contra zeros/inf antes do log10
    Ganho_sem_Raman = np.clip(Ganho_sem_Raman, 1e-300, np.inf)
    Ganho_sem_RamandB = 10 * np.log10(Ganho_sem_Raman)

    GA_sinal = np.exp(expo_arg_clipped)
    GA_sinal = np.clip(GA_sinal, 1e-300, np.inf)
    GA_sinaldB = 10 * np.log10(GA_sinal)

    Ganho_on_off = GA_sinaldB - Ganho_sem_RamandB

    # Armazenamento dos resultados da iteração (usar nan-aware ops)
    ripple_val = np.nan
    if np.any(~np.isnan(Ganho_on_off)):
        ripple_val = np.nanmax(Ganho_on_off) - np.nanmin(Ganho_on_off)

    ripple_arr[i] = ripple_val
    ganho_medio_arr[i] = np.nanmean(GA_sinaldB)
    ganho_on_off_medio_arr[i] = np.nanmean(Ganho_on_off)

# --- PLOTAGEM DOS RESULTADOS ---

# Gráfico 1: Nuvem de Pontos Completa
plt.figure(1)
plt.plot(ripple_arr, ganho_medio_arr, '*')
plt.xlabel('Ripple [dB]')
plt.ylabel('Ganho Médio [dB]')
plt.title('Nuvem de Pontos: Ganho vs. Ripple (Todos os Casos)')
plt.grid(True)

if np.any(ganho_medio_arr):
    y_line = np.linspace(np.min(ganho_medio_arr), np.max(ganho_medio_arr), 100)
    x_line = np.ones_like(y_line) * ripple_bom
    plt.plot(x_line, y_line, 'r--', label=f'Ripple Limite ({ripple_bom} dB)')

plt.legend()


# Gráfico 2: Apenas os Bons Casos
plt.figure(2)
# Encontra os índices onde o ripple é menor ou igual ao limite
bons_indices = np.where(ripple_arr <= ripple_bom)[0]

if len(bons_indices) > 0:
    ripple_bons = ripple_arr[bons_indices]
    ganho_bons = ganho_medio_arr[bons_indices]

    plt.plot(ripple_bons, ganho_bons, '*')
    
    for i in bons_indices:
        plt.text(ripple_arr[i], ganho_medio_arr[i], f' {i}', fontsize=8)
else:
    plt.text(0.5, 0.5, 'Nenhum caso com ripple <= ' + str(ripple_bom) + ' dB encontrado', 
             horizontalalignment='center', verticalalignment='center')


plt.xlabel('Ripple [dB]')
plt.ylabel('Ganho Médio [dB]')
plt.title(f'Casos com Ripple menor que {ripple_bom} dB')
plt.grid(True)


# Exibe os dois gráficos
plt.show()



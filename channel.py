import numpy as np
import math
from scipy import optimize, special
import scipy.integrate as integrate



# FSO channel
fso_path_coe = {'clear_air':0.43e-3, 'haze': 4.3e-3, 'light_fog': 20e-3, 'moderate_fog': 42.2e-3, 'heavy_fog':125e-3}
fso_transmission_wavelength = 1550  # nm
fso_C_0_2 = 1e-14  # m^(-2/3)
wind_speed = 20 # m
fso_bandwidth = 200  # MHz
# fso_noise =  -174 + np.log10(fso_bandwidth * 10^6)  # dBm
fso_noise = -30# dBm/Hz

# IRS parameters
a_l = 0.1 # m, IRS radius
a_r = 0.1 # m, receiver radius
theta_i = np.pi / 4 # incident angle
r_p = 0 # distance from len center to the beam footprint center
# Other Parametter
uav_pos = [220,220,100]
vehicle_nums = 5
cars_height = 2 # m
uav_height = 100 # m
light_speed = 3e8 # tốc độ ánh sáng
target_BER = 10 ** -6


def get_fso_capacity(tx_power: np.ndarray, divergence_angle: np.ndarray, distance: np.ndarray, car_pos: np.ndarray) -> np.ndarray:
    
    # CHANNEL MODEL UAV-vehicle
    shape = distance.shape
    # 1. Atmospheric attenuation (h_a)
    h_a = 10 ** (-fso_path_coe['light_fog'] * distance * 0.1)

    # 2. Atmospheric turbulence Fading (h_f)
    # Calculate Zenith angle (phi_beam_z)
    r_dis = np.linalg.norm(car_pos[:, 0:-1] - uav_pos[0:-1], axis=1)
    phi_beam_z = np.arctan(r_dis / (uav_pos[-1] - cars_height))
    # C_n_2 = fso_C_0_2 * np.exp(-uav_pos[-1] / 100)  # m^(-2/3)
    k = (2 * np.pi) / fso_transmission_wavelength
    sec = 1 / np.cos(phi_beam_z)
    def integrand (h):
        return (0.00594 * (wind_speed / 27) ** 2 * (h * 10 ** -5) ** 10 * np.exp(-h / 1000) + \
                2.7 * 10 ** -16 * np.exp(-h / 1500) + fso_C_0_2 * np.exp(-h / 100)) * \
                    (h - cars_height) ** (5/6)
    integral_value,_ = integrate.quad(integrand, cars_height, uav_height)
    sigma_Rytov = 2.25 * (k ** (7/6)) * (sec ** (11/6)) * integral_value
    # Tính h_f
    h_f = np.random.lognormal(mean=-2*sigma_Rytov, sigma=2*np.sqrt(sigma_Rytov), size=shape)
    
    # 3. Pointing errol (h_p)
    w_0 = 2*fso_transmission_wavelength / (np.pi * divergence_angle) # beam waist at distance = 0
    z_0 = np.pi * w_0**2 / fso_transmission_wavelength # Rayleigh range
    beam_width_z = w_0 * np.sqrt(1 + (distance / z_0) ** 2) # beam width at distance z
    v_p = np.sqrt(np.pi)*a_l / np.sqrt(2) * beam_width_z
    w_eq = beam_width_z**2 * np.sqrt(np.pi)*special.erf(v_p) / (2*v_p*np.exp(-v_p**2))
    A_0 = (special.erf(v_p))**2
    h_p = A_0 * np.exp(-2 * (r_p ** 2) / w_eq ** 2)

    # Channel coefficient (h_1)
    h_1 = h_a * h_f * h_p

    # CAPACITY
    P_irs = 1/np.sqrt(beam_width_z) * special.erf(np.sqrt(2)*theta_i*a_r / beam_width_z)
    P_R = tx_power * h_1 * P_irs # Vehicle's received power
    snr = (10 ** ((P_R - fso_noise) / 10)) * h_1**2
    rate = fso_bandwidth * np.log2(1 + snr)
    return rate


def calculate_irs_received_power(tx_power_dBm, L1, theta_i, w_0):
    """
    Calculate received power at IRS (gắn trên UAV) - Eq (12)
    
    Args:
        tx_power_dBm: Transmit power in dBm (công suất phát từ HAP)
        L1: Distance from HAP to UAV 
        theta_i: Incident angle (góc tới)
        w_0: Beam waist radius (bán kính thắt chùm tia)
    """
    tx_power = 10 ** ((tx_power_dBm - 30) / 10)  # Watt
    wavelength = fso_transmission_wavelength
    
    # Beam width at distance L1 (bề rộng chùm tia tại khoảng cách L1)
    z_0 = np.pi * w_0**2 / wavelength
    w_L1 = w_0 * np.sqrt(1 + (L1 / z_0) ** 2)
    
    # Received power at IRS - Eq (12)
    # Công suất thu tại IRS (gắn trên UAV)
    P_IRS = tx_power * (1 / np.sqrt(w_L1)) * special.erf(
        (np.sqrt(2) * np.cos(theta_i) * a_r) / w_L1
    )
    
    return P_IRS


def calculate_optimal_divergence(L2, a_l, wavelength):
    """
    Calculate optimal divergence angle for phase shift - Eq (17)
    Giải phương trình: (L2^2 * theta^4)/16 - (a_l^4 * theta^2) = lambda^2/pi^2
    
    Args:
        L2: Distance from IRS to vehicle (khoảng cách từ IRS đến xe)
        a_l: Receiver lens radius (bán kính thấu kính thu)
        wavelength: FSO wavelength (bước sóng)
    
    Returns:
        theta_opt: Optimal divergence angle (góc phân kỳ tối ưu)
    """
    # Chuyển thành phương trình bậc 2 theo X = theta^2
    # (L2^2/16)*X^2 - a_l^4*X - lambda^2/pi^2 = 0
    
    A = L2**2 / 16
    B = -a_l**4
    C = -wavelength**2 / np.pi**2
    
    # Giải phương trình bậc 2
    discriminant = B**2 - 4*A*C
    if discriminant >= 0:
        X = (-B + np.sqrt(discriminant)) / (2*A)  # Lấy nghiệm dương
        if X > 0:
            theta_opt = np.sqrt(X)
            return theta_opt
    
    # Default nếu không có nghiệm
    return 0.01  # rad


def calculate_phase_shift_profile(y, L1, L2, theta_i, theta_r, w_0, wavelength):
    """
    Calculate phase shift profile for IRS - Eq (15)
    
    Args:
        y: Position on IRS (vị trí trên bề mặt IRS)
        L1: Distance from HAP to IRS
        L2: Distance from IRS to vehicle (khoảng cách từ IRS đến xe)
        theta_i: Incident angle (góc tới)
        theta_r: Reflection angle (góc phản xạ)
        w_0: Beam waist radius
        wavelength: FSO wavelength
    
    Returns:
        delta_psi: Phase shift at position y (độ dịch pha tại vị trí y)
    """
    k = 2 * np.pi / wavelength
    z_0 = np.pi * w_0**2 / wavelength
    
    # Phase of incoming beam tại vị trí y trên IRS
    z1 = L1 + y * np.sin(theta_i)
    R1 = z1 * (1 + (z_0 / z1) ** 2)  # Curvature radius
    zeta1 = np.arctan(z1 / z_0)  # Gouy phase
    psi_in = -k * z1 - k * (y * np.cos(theta_i)) ** 2 / (2 * R1) + zeta1
    
    # Calculate virtual beam waist for reflected beam
    w_irs = w_0 * np.sqrt(1 + (L1 / z_0) ** 2) / np.cos(theta_i)
    w_eq_r = w_irs * np.cos(theta_r)
    
    # Beam waist of virtual beam
    w0_tilde = np.sqrt(w_eq_r**2 / 2 - np.sqrt(np.pi**2 * w_eq_r**4 - 4 * L2**2 * wavelength**2) / (2 * np.pi))
    
    # Rayleigh range for virtual beam
    z0_tilde = np.pi * w0_tilde**2 / wavelength
    
    # Phase of virtual beam (with reversed direction)
    z2 = -(L2 + y * np.sin(theta_r))
    R2 = z2 * (1 + (z0_tilde / z2) ** 2)
    zeta2 = np.arctan(z2 / z0_tilde)
    psi_virtual = -k * z2 - k * (y * np.cos(theta_r)) ** 2 / (2 * R2) + zeta2
    
    # Phase shift profile - Eq (15)
    delta_psi = np.pi - psi_in + psi_virtual
    
    return delta_psi
import psychrolib as psy
import numpy as np
import math

class HVACSimulation:
    def __init__(self,
                 # Room Config (Fixed)
                 dt=600, P_air=101325, L_room=5, W_room=4, H_room=2.5, A_window=4,
                 # AHU Config (Fixed)
                 Max_Air_FR=0.1, P_fan=5.0, Max_Vent_FR=0.1, Max_Cooling_PW=2500, Max_Heating_PW=1000,
                 # Standard Definitions (Fixed)
                 Q_S_Per_Occ=200 * 0.293, Q_L_Per_Occ=180 * 0.293, Q_L_Other=0,
                 U_wall=0.284, U_window=1.987,
                 # Constants (Fixed)
                 h_fg_room=2441000, Cp_Air=1005):
        # Room Config
        self.dt = dt  # s Time to simulate
        self.P_air = P_air  # Pa - Atmospheric pressure (sea level)
        self.L_room = L_room  # m - Length of the room
        self.W_room = W_room  # m - Width of the room
        self.H_room = H_room  # m - Height of the room
        self.A_window = A_window  # m^2 - Area of the window
        self.A_wall = (L_room + W_room) * H_room - A_window  # m^2 - Area of the wall
        self.V_room = L_room * W_room * H_room  # m^3 - Volume of the room

        # AHU Config
        self.Max_Air_FR = Max_Air_FR  # m^3/s Maximum airflow rate
        self.P_fan = P_fan  # W Fan power
        self.Max_Vent_FR = Max_Vent_FR  # m^3/s Maximum ventilation airflow rate
        self.Max_Cooling_PW = Max_Cooling_PW  # W Maximum Cooling load
        self.Max_Heating_PW = Max_Heating_PW  # W Maximum heating load

        # Standard Definitions
        self.Q_S_Per_Occ = Q_S_Per_Occ  # W - Sensible Heat from Occupants
        self.Q_L_Per_Occ = Q_L_Per_Occ  # W - Latent Heat from Occupants
        self.Q_L_Other = Q_L_Other  # W - Other latent heat sources
        self.U_wall = U_wall  # W/m^2.K
        self.U_window = U_window  # W/m^2.K

        # Constants
        self.h_fg_room = h_fg_room  # J/kg
        self.Cp_Air = Cp_Air  # J/kg-K

    def calculate_room_conditions(self, T_room, RH_room, T_out, RH_out):
        """Calculate initial room conditions."""
        self.W_room = psy.GetHumRatioFromRelHum(T_room, max(RH_room, 0), self.P_air)  # kg water/kg dry air
        self.h_room = psy.GetMoistAirEnthalpy(T_room, self.W_room)  # J/kg
        self.p_air = psy.GetMoistAirDensity(T_room, self.W_room, self.P_air)  # kg/m^3
        self.W_out = psy.GetHumRatioFromRelHum(T_out, RH_out, self.P_air)  # kg water/kg dry air

    def calculate_calculated_parameters(self, Control_Air_AF, Control_Vent_AF, Control_Cooling_PW, Control_Heating_PW):
        """Calculate airflow and power parameters."""
        self.Air_AF = self.Max_Air_FR * Control_Air_AF  # m^3/s
        self.Vent_AF = self.Max_Air_FR * Control_Vent_AF  # m^3/s
        self.Cooling_PW = self.Max_Cooling_PW * Control_Cooling_PW  # W
        self.Heating_PW = self.Max_Heating_PW * Control_Heating_PW  # W

    def calculate_heat_loads(self, N_Occ, Q_Equ, T_room, T_out):
        """Calculate total sensible and latent heat loads."""
        self.Q_S_envelope = self.U_wall * self.A_wall * (T_out - T_room) + \
                           self.U_window * self.A_window * (T_out - T_room)  # W
        self.Q_S_total = self.Q_S_Per_Occ * N_Occ + Q_Equ + self.Q_S_envelope  # W
        self.Q_L_total = self.Q_L_Per_Occ * N_Occ + self.Q_L_Other  # W

    def calculate_mixed_air(self):
        """Calculate mixed air conditions."""
        self.T_MA = (self.Vent_AF * self.T_out + (self.Air_AF - self.Vent_AF) * self.T_room) / self.Air_AF
        self.W_MA = (self.Vent_AF * self.W_out + (self.Air_AF - self.Vent_AF) * self.W_room) / self.Air_AF
        self.RH_MA = psy.GetRelHumFromHumRatio(self.T_MA, max(self.W_MA, 0), self.P_air)
        self.h_MA = psy.GetMoistAirEnthalpy(self.T_MA, max(self.W_MA, 0))  # J/kg

    def calculate_cooling_coil(self):
        """Simulate cooling coil process."""
        self.h_Coil = self.h_MA - (self.Cooling_PW / (self.p_air * self.Air_AF))
        RH_Coil = 1.0  # Assume saturation
        T_min, T_max = 0.0, 60.0  # °C
        T_step = 0.01
        tolerance = 100  # J/kg

        self.T_Coil = None
        self.W_Coil = None
        for T in np.arange(T_min, T_max, T_step):
            W = psy.GetHumRatioFromRelHum(T, RH_Coil, self.P_air)
            h = psy.GetMoistAirEnthalpy(T, W)
            if abs(h - self.h_Coil) < tolerance:
                self.T_Coil = T
                self.W_Coil = W
                break
        self.RH_Coil = psy.GetRelHumFromHumRatio(self.T_Coil, max(self.W_Coil, 0), self.P_air)
        self.T_dew = psy.GetTDewPointFromRelHum(self.T_Coil, self.RH_Coil)

    def calculate_fan_heat(self):
        """Calculate fan heat addition."""
        self.ΔT_fan = self.P_fan / (self.p_air * self.Cp_Air * self.Air_AF)
        self.T_fan = self.T_Coil + self.ΔT_fan
        self.W_fan = self.W_Coil
        self.RH_fan = psy.GetRelHumFromHumRatio(self.T_fan, max(self.W_fan, 0), self.P_air)
        self.h_fan = psy.GetMoistAirEnthalpy(self.T_fan, self.W_fan)

    def calculate_heating_coil(self):
        """Simulate heating coil process."""
        self.T_Out = self.T_fan + (self.Heating_PW / (self.p_air * self.Cp_Air * self.Air_AF))
        self.W_Out = self.W_fan
        self.RH_Out = psy.GetRelHumFromHumRatio(self.T_Out, max(self.W_Out, 0), self.P_air)
        self.h_Out = psy.GetMoistAirEnthalpy(self.T_Out, max(self.W_Out, 0))  # J/kg

    def calculate_final_conditions(self):
        """Calculate final room air conditions."""
        self.T_final = (self.T_Out + self.Q_S_total / (self.p_air * self.Cp_Air * self.Air_AF) +
                       (self.T_room - self.T_Out - (self.Q_S_total / (self.p_air * self.Cp_Air * self.Air_AF))) *
                       math.exp(-(self.Air_AF / self.V_room) * self.dt))
        self.W_final = (self.W_Out + self.Q_L_total / (self.p_air * self.h_fg_room * self.Air_AF) +
                       (self.W_room - self.W_Out - (self.Q_L_total / (self.p_air * self.h_fg_room * self.Air_AF))) *
                       math.exp(-(self.Air_AF / self.V_room) * self.dt))
        self.RH_final = psy.GetRelHumFromHumRatio(self.T_final, max(self.W_final, 0), self.P_air)
        self.h_final = psy.GetMoistAirEnthalpy(self.T_final, max(self.W_final, 0))  # J/kg

    def calculate_energy_consumption(self):
        """Calculate energy consumption."""
        self.E_fan = self.P_fan * self.dt  # J
        self.E_cooling = self.Cooling_PW * self.dt  # J
        self.E_heating = self.Heating_PW * self.dt  # J
        self.E_fan_kWh = self.E_fan / 3.6e6
        self.E_cooling_kWh = self.E_cooling / 3.6e6
        self.E_heating_kWh = self.E_heating / 3.6e6
        self.E_total_kWh = self.E_fan_kWh + self.E_cooling_kWh + self.E_heating_kWh

    def run_simulation(self, 
                       Control_Air_AF=0.5, Control_Vent_AF=0.1, Control_Cooling_PW=0.9, Control_Heating_PW=0.1,
                       N_Occ=3, Q_Equ=500, T_room=25, RH_room=0.5, T_out=35, RH_out=0.7):
        """Run the full simulation and return results as a dictionary."""
        # Store input parameters
        self.Control_Air_AF = Control_Air_AF
        self.Control_Vent_AF = Control_Vent_AF
        self.Control_Cooling_PW = Control_Cooling_PW
        self.Control_Heating_PW = Control_Heating_PW
        self.N_Occ = N_Occ
        self.Q_Equ = Q_Equ
        self.T_room = T_room
        self.RH_room = RH_room
        self.T_out = T_out
        self.RH_out = RH_out

        # Run calculations
        self.calculate_room_conditions(T_room, RH_room, T_out, RH_out)
        self.calculate_calculated_parameters(Control_Air_AF, Control_Vent_AF, Control_Cooling_PW, Control_Heating_PW)
        self.calculate_heat_loads(N_Occ, Q_Equ, T_room, T_out)
        self.calculate_mixed_air()
        self.calculate_cooling_coil()
        self.calculate_fan_heat()
        self.calculate_heating_coil()
        self.calculate_final_conditions()
        self.calculate_energy_consumption()

        # Organize results
        results = {
            # Top-level: Input parameters and key results
            'Control_Air_AF': self.Control_Air_AF,
            'Control_Vent_AF': self.Control_Vent_AF,
            'Control_Cooling_PW': self.Control_Cooling_PW,
            'Control_Heating_PW': self.Control_Heating_PW,
            'N_Occ': self.N_Occ,
            'Q_Equ': self.Q_Equ,
            'T_room': self.T_room,
            'RH_room': self.RH_room,
            'T_out': self.T_out,
            'RH_out': self.RH_out,
            'T_final': self.T_final,
            'W_final': self.W_final,
            'RH_final': self.RH_final,
            'h_final': self.h_final,
            'E_total_kWh': self.E_total_kWh,
            # Secondary-level: Intermediate calculations
            'secondary_data': {
                'Q_S_Per_Occ': self.Q_S_Per_Occ,
                'Q_L_Per_Occ': self.Q_L_Per_Occ,
                'W_room': self.W_room,
                'h_room': self.h_room,
                'h_fg_room': self.h_fg_room,
                'p_air': self.p_air,
                'Cp_Air': self.Cp_Air,
                'Q_S_envelope': self.Q_S_envelope,
                'Q_S_total': self.Q_S_total,
                'Q_L_total': self.Q_L_total,
                'T_MA': self.T_MA,
                'W_MA': self.W_MA,
                'RH_MA': self.RH_MA,
                'h_MA': self.h_MA,
                'T_Coil': self.T_Coil,
                'W_Coil': self.W_Coil,
                'RH_Coil': self.RH_Coil,
                'h_Coil': self.h_Coil,
                'T_dew': self.T_dew,
                'ΔT_fan': self.ΔT_fan,
                'T_fan': self.T_fan,
                'W_fan': self.W_fan,
                'RH_fan': self.RH_fan,
                'h_fan': self.h_fan,
                'T_Out': self.T_Out,
                'W_Out': self.W_Out,
                'RH_Out': self.RH_Out,
                'h_Out': self.h_Out,
                'E_fan_kWh': self.E_fan_kWh,
                'E_cooling_kWh': self.E_cooling_kWh,
                'E_heating_kWh': self.E_heating_kWh
            }
        }
        return results

# if __name__ == "__main__":
#     sim = HVACSimulation()
#     results = sim.run_simulation()
#     # Example of accessing results
#     print("Top-level results:")
#     for key, value in results.items():
#         if key != 'secondary_data':
#             print(f"{key}: {value}")
#     print("\nSecondary data:")
#     for key, value in results['secondary_data'].items():
#         print(f"{key}: {value}")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

class LIFNeuron:
    """
    Leaky Integrate-and-Fire (LIF) Neuron Model.
    This is the foundational model used in most Neuromorphic Hardware
    (like Intel Loihi, IBM TrueNorth, and SpiNNaker).
    """
    def __init__(self, tau_m=10.0, v_rest=-50.0, v_th=-45.0, v_reset=-50.0, R=1.0):
        self.tau_m = tau_m        # Membrane time constant (ms)
        self.v_rest = v_rest      # Resting membrane potential (mV)
        self.v_th = v_th          # Spike threshold (mV)
        self.v_reset = v_reset    # Reset potential after a spike (mV)
        self.R = R                # Membrane resistance (MOhm)

    def step(self, v_prev, I_inj, dt):
        """
        Simulate a single time step. 
        Instead of processing a whole array at once, we step through time
        so we can connect neurons together dynamically.
        """
        # Leaky Integrate-and-Fire differential equation calculation
        dv = (-(v_prev - self.v_rest) + self.R * I_inj) * (dt / self.tau_m)
        v_new = v_prev + dv
        spike = False
        
        # Check for action potential (spike)
        if v_new >= self.v_th:
            v_new = self.v_reset
            spike = True     # Record the spike
            
        return v_new, spike


def run_interactive_sim():
    # 1. Setup Simulation Time
    T = 100.0  # Total simulation time in ms
    dt = 0.1   # Time step in ms
    time_steps = int(T / dt)
    time_axis = np.linspace(0, T, time_steps)
    
    # Initialize default parameters
    base_current_A = 16.0
    synapse_weight = 30.0  # How strong the signal from A to B is
    tau_synapse = 5.0      # How long the neurotransmitters last in the gap
    
    # 2. Instantiate our Neuromorphic Network (2 Neurons)
    neuron_A = LIFNeuron(tau_m=10.0) # Fast learning neuron
    neuron_B = LIFNeuron(tau_m=15.0) # Slower, needs more charge to fire
    
    def simulate_network(curr_A, weight, tau_syn):
        v_A = np.zeros(time_steps)
        v_B = np.zeros(time_steps)
        I_syn = np.zeros(time_steps) # The chemical signal between A and B
        
        v_A[0] = neuron_A.v_rest
        v_B[0] = neuron_B.v_rest
        
        # Create input current for Neuron A (with biological noise)
        I_inj_A = curr_A + 3.0 * np.random.randn(time_steps)
        
        for t in range(1, time_steps):
            # STEP 1: Process Neuron A
            v_A[t], spike_A = neuron_A.step(v_A[t-1], I_inj_A[t], dt)
            if spike_A:
                v_A[t-1] = 20.0  # Visual trick: draw the spike peak
                
            # STEP 2: Process the Synapse (Connection from A to B)
            # The current leaks away naturally over time...
            dI_syn = -I_syn[t-1] * (dt / tau_syn)
            # ...but jumps up instantly if Neuron A spikes!
            I_syn[t] = I_syn[t-1] + dI_syn + (weight if spike_A else 0.0)
            
            # STEP 3: Process Neuron B
            # Neuron B's ONLY input is the synaptic current coming from A
            v_B[t], spike_B = neuron_B.step(v_B[t-1], I_syn[t], dt)
            if spike_B:
                v_B[t-1] = 20.0  # Visual trick: draw the spike peak
                
        return v_A, I_syn, v_B

    # Run initial simulation
    v_A, I_syn, v_B = simulate_network(base_current_A, synapse_weight, tau_synapse)
    
    # 3. Setup Interactive Visualization
    fig, (ax_vA, ax_syn, ax_vB) = plt.subplots(3, 1, figsize=(10, 9))
    plt.subplots_adjust(bottom=0.35, hspace=0.5) 
    
    # Neuron A Plot
    ax_vA.set_title("Neuron A (Pre-Synaptic) - Receives External Stimulus")
    ax_vA.set_ylabel("Voltage (mV)")
    ax_vA.set_xlim(0, T)
    ax_vA.set_ylim(-80, 30)
    line_vA, = ax_vA.plot(time_axis, v_A, color='blue')
    ax_vA.axhline(neuron_A.v_th, color='red', linestyle='--', alpha=0.5)
    
    # Synapse Plot
    ax_syn.set_title("Synaptic Current (Neurotransmitters from A to B)")
    ax_syn.set_ylabel("Current (nA)")
    ax_syn.set_xlim(0, T)
    line_syn, = ax_syn.plot(time_axis, I_syn, color='green')
    
    # Neuron B Plot
    ax_vB.set_title("Neuron B (Post-Synaptic) - Only spikes when A passes enough charge")
    ax_vB.set_xlabel("Time (ms)")
    ax_vB.set_ylabel("Voltage (mV)")
    ax_vB.set_xlim(0, T)
    ax_vB.set_ylim(-80, 30)
    line_vB, = ax_vB.plot(time_axis, v_B, color='purple')
    ax_vB.axhline(neuron_B.v_th, color='red', linestyle='--', alpha=0.5)
    
    # 4. Define UI Sliders
    ax_currA = plt.axes([0.15, 0.2, 0.7, 0.03])
    ax_weight = plt.axes([0.15, 0.15, 0.7, 0.03])
    ax_tau_syn = plt.axes([0.15, 0.1, 0.7, 0.03])
    
    slider_currA = Slider(ax_currA, 'Stimulus to A', 0.0, 40.0, valinit=base_current_A)
    slider_weight = Slider(ax_weight, 'Synapse Weight', 0.0, 60.0, valinit=synapse_weight)
    slider_tau_syn = Slider(ax_tau_syn, 'Synapse Duration', 1.0, 20.0, valinit=tau_synapse)
    
    # 5. Define Update Function for Real-Time Feedback
    def update(val):
        curr = slider_currA.val
        weight = slider_weight.val
        tau = slider_tau_syn.val
        
        # Re-simulate the network
        new_vA, new_Isyn, new_vB = simulate_network(curr, weight, tau)
        
        # Update the plots
        line_vA.set_ydata(new_vA)
        line_syn.set_ydata(new_Isyn)
        line_vB.set_ydata(new_vB)
        
        # Auto-scale Y-axis for the current
        ax_syn.set_ylim(min(new_Isyn)-1, max(new_Isyn)+5)
        
        fig.canvas.draw_idle()

    # Connect the sliders
    slider_currA.on_changed(update)
    slider_weight.on_changed(update)
    slider_tau_syn.on_changed(update)
    
    plt.show()

if __name__ == "__main__":
    run_interactive_sim()
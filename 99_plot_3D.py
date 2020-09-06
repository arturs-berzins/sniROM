"""
Creates a 3D plot of learnt manifolds over features.
Very useful for building intuition of the dataset and models.
All features and targets are standardized.
"""
import config
import utils
import numpy as np
from matplotlib import pyplot, cm
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D

# Plot along these two parameters:
parameter_indices_to_plot = [0,1]
# Use this model:
model_key = 'FNN'

def main():
    parameters = {}
    parameters['axes'] = parameter_indices_to_plot
    assert len(parameters['axes']) == 2, \
        'Plot_3D can only plot over 2 parameters.'
    assert all(0<=p<=config.P-1 for p in parameters['axes']), \
        'Provided parameters do not coincide with those in config.'
    parameters['sliders'] = list(set(range(config.P))-set(parameters['axes']))
        
    # Store references to plots, otherwise the widgets become unresponsive due
    # to garbage collector. https://stackoverflow.com/a/42884505
    plots = {}

    scaler = utils.load_scaler()
    for component in config.components:
        model_constructor = utils.models[model_key]
        model = model_constructor()
        model.load(utils.model_dir, component)
        
        # Initialize and load data structres
        features = {}; targets = {}; outputs = {}
        for dataset in ['train', 'test']:
            features[dataset] = utils.load_features(dataset)
            targets[dataset] = utils.load_targets(dataset, component)
            outputs[dataset] = None
        # Create the interactive 3D plot
        plots[component] = Plot_3D(component, outputs, targets, features, model, parameters, scaler)
    pyplot.show()

class Plot_3D:
    def __init__(self, component, outputs, targets, features, model, parameters, scaler):
        # Copy the arguments
        self.component = component
        self.outputs = outputs
        self.targets = targets
        self.features = features
        self.model = model
        self.parameters = parameters
        self.scaler = scaler
        # Get axes' parameter indices
        self.p1 = parameters['axes'][0]
        self.p2 = parameters['axes'][1]
        # these will all basically be +-1.73 due to nature of StandardScaler
        self.feature_range = self.scaler.transform(
            np.array(list(config.mu_range.values())).T
            ).T
    
        self.fig = pyplot.figure()
        self.ax1 = self.fig.add_subplot(111, projection='3d', proj_type = 'ortho')
        self.fig.suptitle('Learnt manifold for component ' + self.component)

        # Create axes for bases sliders
        axmax = pyplot.axes([0.25, 0.10, 0.65, 0.03])
        axmin = pyplot.axes([0.25, 0.05, 0.65, 0.03])
        self.L = config.num_basis[component]
        # Arguments: name, min, max, init, step, format
        self.slider_l_min = Slider(axmin, 'basis min', 0, self.L-1, valinit=0,
            valstep=1, valfmt='%d')
        self.slider_l_max = Slider(axmax, 'basis max', 0, self.L-1, valinit=0,
            valstep=1, valfmt='%d')
        # Register event handlers
        self.slider_l_min.on_changed(self.react_to_basis_slider)
        self.slider_l_max.on_changed(self.react_to_basis_slider)
        
        # Create sliders for sliders' parameters
        self.sliders_for_parameters = {}
        for i, idx in enumerate(self.parameters['sliders']):
            ax = pyplot.axes([0.25, 0.15 + 0.05*i, 0.65, 0.03])
            min_val = self.feature_range[idx][0]
            max_val = self.feature_range[idx][1]
            valinit = 0.5*(min_val + max_val)
            # Arguments: axis, name, min, max, init
            self.sliders_for_parameters[idx]    \
                = Slider(ax, config.mu_names[idx], min_val, max_val, valinit=valinit)
            # Register event handler
            self.sliders_for_parameters[idx]    \
                .on_changed(self.react_to_parameter_slider)
        
        # Create an XY grid from the axes features
        Nx = 25
        Ny = 25
        x = np.linspace(*self.feature_range[self.p1], Nx)
        y = np.linspace(*self.feature_range[self.p2], Ny)
        self.X, self.Y = np.meshgrid(x, y)
        
        # Initialize an array holding features from all axes and sliders
        self.features['manifold'] = np.zeros([Nx*Ny, config.P])
        # Insert axes' features into manifold feature array
        self.features['manifold'][:,self.p1] = np.ravel(self.X)
        self.features['manifold'][:,self.p2] = np.ravel(self.Y)
        
        # Define list of colors each basis
        cmap = pyplot.get_cmap('tab10')
        self.colors = [cmap(l%10) for l in range(0, self.L)]
        
        # For initial plot
        self.react_to_parameter_slider(None)
    
    def react_to_parameter_slider(self, val):
        # Insert sliders' current values into manifold feature array
        for idx in self.parameters['sliders']:
            current_value_on_slider = self.sliders_for_parameters[idx].val
            self.features['manifold'][:,idx] = current_value_on_slider
        
        # Evaluate the manifold
        self.outputs['manifold'] = self.model.evaluate(self.features['manifold'])
        
        # Call the actual plotting routine
        self.react_to_basis_slider(None)
    
    def react_to_basis_slider(self, val):
        # Clear current
        self.ax1.cla()
        
        # Loop through all selected bases
        for l in range(int(self.slider_l_min.val), int(self.slider_l_max.val)+1):
            # Get color
            c = self.colors[l]
            
            ## Scatter learnt manifold
            Z = self.outputs['manifold'][:,l].reshape(self.X.shape)
            self.ax1.plot_surface(self.X, self.Y, Z, alpha = 0.8, color=c, shade=True)
            
            ## Scatter train set
            x = self.features['train'][:,self.p1]
            y = self.features['train'][:,self.p2]
            z = self.targets['train'][:,l]
            self.ax1.scatter(x, y, z, color=c, label=f"training l={l}")
            
            ## Scatter test set
            x = self.features['test'][:,self.p1]
            y = self.features['test'][:,self.p2]
            z = self.targets['test'][:,l]
            self.ax1.scatter(x, y, z, color=c, marker="X", label=f"test l={l}")
            
        self.ax1.set_xlabel(config.mu_names[self.p1])
        self.ax1.set_ylabel(config.mu_names[self.p2])
        self.ax1.set_xlim(self.feature_range[self.p1])
        self.ax1.set_ylim(self.feature_range[self.p2])
        self.ax1.legend()
        self.fig.subplots_adjust(bottom=0.05*(len(self.parameters['sliders']) + 2), top=0.9)
        self.fig.canvas.draw_idle()

if __name__ == '__main__':
    main()


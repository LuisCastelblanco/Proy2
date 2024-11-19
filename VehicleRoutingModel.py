from pyomo.environ import *
import pandas as pd
from itertools import product
import matplotlib.pyplot as plt

class VehicleRoutingModel:
    def __init__(self, client_data, vehicle_data, distance_data):
        """
        Inicializa el modelo con los datos procesados.
        """
        # Identificar Depósitos y Clientes
        self.depots = client_data['DepotID'].unique().tolist()
        self.clients = [cid for cid in client_data['ClientID'] if cid not in self.depots]
        self.locations = self.depots + self.clients

        # Asignar Identificadores Únicos a Vehículos
        self.vehicle_ids = ['V{}'.format(i+1) for i in range(len(vehicle_data))]
        self.vehicle_type = dict(zip(self.vehicle_ids, vehicle_data['VehicleType']))
        self.capacity = dict(zip(self.vehicle_ids, vehicle_data['Capacity']))
        self.vehicle_range = dict(zip(self.vehicle_ids, vehicle_data['Range']))

        # Demanda de Clientes (Asumimos que 'Product' es Demanda en kg)
        self.demand = dict(zip(client_data['ClientID'], client_data['Product']))
        # Demanda de Depósitos es 0
        for depot in self.depots:
            self.demand[depot] = 0

        # Coordenadas de Todas las Ubicaciones
        coords = client_data[['ClientID', 'Longitude', 'Latitude']].set_index('ClientID')
        self.coordinates = coords.loc[self.locations].to_dict('index')

        # Matriz de Distancias
        self.distances = distance_data

        # Crear Modelo
        self.model = ConcreteModel()
        self.data = {
            'clients': self.clients,
            'depots': self.depots,
            'locations': self.locations,
            'vehicles': self.vehicle_ids,
            'vehicle_type': self.vehicle_type,
            'demand': self.demand,
            'capacity': self.capacity,
            'range': self.vehicle_range,
            'distances': self.distances,
        }

    def build_model(self):
        """
        Construye el modelo matemático basado en Pyomo.
        """
        model = self.model
        data = self.data

        # Conjuntos
        model.N = Set(initialize=data['clients'])    # Clientes
        model.D = Set(initialize=data['depots'])     # Depósitos
        model.L = Set(initialize=data['locations'])  # Todas las ubicaciones
        model.V = Set(initialize=data['vehicles'])   # Vehículos
        model.E = Set(within=model.L * model.L, initialize=data['distances'].keys())  # Arcos (i, j)

        # Parámetros
        model.demand = Param(model.L, initialize=data['demand'], default=0)  # Demanda de clientes y depots
        model.dist = Param(model.E, initialize=data['distances'])            # Distancias
        model.capacity = Param(model.V, initialize=data['capacity'], default=0)  # Capacidad de vehículos
        model.vehicle_range = Param(model.V, initialize=data['range'], default=0)  # Rango por vehículo

        # Variables de decisión
        model.x = Var(model.V, model.E, domain=Binary)  # Si vehículo v usa arco (i, j)
        model.y = Var(model.V, model.L, domain=Binary)  # Si vehículo v atiende cliente i
        model.z = Var(model.V, domain=Binary)           # Si vehículo v se usa
        model.u = Var(model.V, model.L, bounds=(0, None))  # Variables para eliminación de subtours

        # Función objetivo: Minimizar la distancia total recorrida
        def objective_rule(model):
            return sum(model.dist[i, j] * model.x[v, i, j]
                       for v in model.V for (i, j) in model.E)
        model.objective = Objective(rule=objective_rule, sense=minimize)

        # Restricciones
        self.add_constraints()

    def add_constraints(self):
        model = self.model

        # Cada cliente es atendido exactamente una vez
        def coverage_rule(model, i):
            return sum(model.y[v, i] for v in model.V) == 1
        model.coverage = Constraint(model.N, rule=coverage_rule)

        # Restricciones de Conservación de Flujo
        def flow_in_rule(model, v, i):
            inflow = sum(model.x[v, j, i] for j in model.L if (j, i) in model.E)
            return inflow == model.y[v, i]
        model.flow_in = Constraint(model.V, model.N, rule=flow_in_rule)

        def flow_out_rule(model, v, i):
            outflow = sum(model.x[v, i, j] for j in model.L if (i, j) in model.E)
            return outflow == model.y[v, i]
        model.flow_out = Constraint(model.V, model.N, rule=flow_out_rule)

        # Salida de los depots
        def depot_start_rule(model, v):
            return sum(model.x[v, d, j] for d in model.D for j in model.L if (d, j) in model.E) == model.z[v]
        model.depot_start = Constraint(model.V, rule=depot_start_rule)

        # Regreso a los depots
        def depot_end_rule(model, v):
            return sum(model.x[v, i, d] for d in model.D for i in model.L if (i, d) in model.E) == model.z[v]
        model.depot_end = Constraint(model.V, rule=depot_end_rule)

        # Capacidad de los vehículos
        def capacity_rule(model, v):
            return sum(model.demand[i] * model.y[v, i] for i in model.N) <= model.capacity[v]
        model.capacity_constraint = Constraint(model.V, rule=capacity_rule)

        # Rango máximo de los vehículos
        def range_rule(model, v):
            return sum(model.dist[i, j] * model.x[v, i, j] for (i, j) in model.E) <= model.vehicle_range[v]
        model.vehicle_range_constraint = Constraint(model.V, rule=range_rule)

        # Eliminación de subtours (Formulación de Miller-Tucker-Zemlin)
        def subtour_rule(model, v, i, j):
            if i not in model.D and j not in model.D and i != j:
                return model.u[v, i] - model.u[v, j] + model.capacity[v] * model.x[v, i, j] <= model.capacity[v] - model.demand[j]
            else:
                return Constraint.Skip
        model.subtour_elimination = Constraint(model.V, model.L, model.L, rule=subtour_rule)

        # Relación entre x e y: Si un vehículo atiende un cliente, debe recorrer al menos un arco que lo visite
        def relation_rule(model, v, i):
            if i in model.D:
                return Constraint.Skip  # No aplicar a depots
            return model.y[v, i] <= sum(model.x[v, i, j] for j in model.L if (i, j) in model.E)
        model.relation = Constraint(model.V, model.L, rule=relation_rule)

        # Uso de vehículos: Si un vehículo no se usa, no puede recorrer ningún arco
        def vehicle_usage_rule(model, v, i, j):
            return model.x[v, i, j] <= model.z[v]
        model.vehicle_usage = Constraint(model.V, model.E, rule=vehicle_usage_rule)

    def solve(self, solver_name='highs'):
        """
        Resuelve el modelo usando el solver especificado.
        """
        solver = SolverFactory(solver_name)
        if not solver.available():
            raise RuntimeError(f"Solver {solver_name} no está disponible. Por favor, instálalo correctamente.")
        result = solver.solve(self.model, tee=True)
        if (result.solver.status != SolverStatus.ok) or (result.solver.termination_condition != TerminationCondition.optimal):
            raise ValueError("El solver no encontró una solución óptima.")
        self.model.solutions.load_from(result)

    def export_routes(self, output_file='routes.csv'):
        """
        Exporta las rutas generadas a un archivo CSV.
        """
        model = self.model
        routes = []

        for v in model.V:
            # Encontrar los arcos utilizados por el vehículo
            arcs = [(i, j) for (i, j) in self.data['distances'].keys() if model.x[v, i, j].value > 0.5]
            if not arcs:
                continue  # Vehículo no utilizado

            # Construir la ruta secuencialmente
            route = []
            # Encontrar el depot de inicio
            start_depot = next((d for d in self.data['depots'] if any(x for x in self.data['distances'].keys() if x[0] == d and x[1] in self.clients)), None)
            if not start_depot:
                start_depot = self.data['depots'][0]  # Asignar el primer depot si no se encuentra uno específico

            current_location = start_depot
            while True:
                next_steps = [j for (i, j) in arcs if i == current_location]
                if not next_steps:
                    break
                next_location = next_steps[0]
                routes.append({'Vehicle': v, 'From': current_location, 'To': next_location, 'Distance': self.data['distances'][(current_location, next_location)]})
                current_location = next_location
                if current_location in self.data['depots']:
                    break

        # Guardar en un archivo CSV
        pd.DataFrame(routes).to_csv(output_file, index=False)
        print(f"Rutas exportadas a {output_file}")

    def plot_routes(self):
        """
        Visualiza las rutas generadas en un gráfico.
        """
        model = self.model
        coords = pd.DataFrame.from_dict(self.coordinates, orient='index')

        plt.figure(figsize=(12, 10))
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']
        color_map = {}

        for idx, v in enumerate(model.V):
            color = colors[idx % len(colors)]
            color_map[v] = color
            for (i, j) in self.data['distances'].keys():
                if model.x[v, i, j].value > 0.5:
                    x_coords = [self.coordinates[i]['Longitude'], self.coordinates[j]['Longitude']]
                    y_coords = [self.coordinates[i]['Latitude'], self.coordinates[j]['Latitude']]
                    plt.plot(x_coords, y_coords, color=color, linewidth=2, label=f'Vehicle {v}' if v not in color_map else "")

        # Dibujar nodos
        plt.scatter(coords['Longitude'], coords['Latitude'], c='black', zorder=5)
        for loc in coords.index:
            plt.text(self.coordinates[loc]['Longitude'] + 0.001, self.coordinates[loc]['Latitude'] + 0.001, str(loc), fontsize=9)

        # Crear leyenda única
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())

        plt.title('Rutas Generadas')
        plt.xlabel('Longitud')
        plt.ylabel('Latitud')
        plt.grid(True)
        plt.show()

def create_distance_matrix(locations_df):
    """
    Crea una matriz de distancias Euclidianas entre todas las ubicaciones.
    """
    distance_matrix = {}
    for i, j in product(locations_df.index, repeat=2):
        if i != j:
            distance = ((locations_df.loc[i, 'Longitude'] - locations_df.loc[j, 'Longitude'])**2 +
                        (locations_df.loc[i, 'Latitude'] - locations_df.loc[j, 'Latitude'])**2)**0.5
            distance_matrix[(i, j)] = distance
    return distance_matrix

# Lectura de datos
clients = pd.read_csv(r'case_1_base\Clients.csv')
vehicles = pd.read_csv(r'case_1_base\multi_vehicles.csv')

# Preprocesamiento de datos
# No es necesario renombrar columnas ya que ya están correctas

# Crear matriz de distancias
locations_df = clients[['ClientID', 'Longitude', 'Latitude']].set_index('ClientID')
distance_matrix = create_distance_matrix(locations_df)

# Crear el modelo
vr_model = VehicleRoutingModel(clients, vehicles, distance_matrix)

# Construir y resolver el modelo
vr_model.build_model()
vr_model.solve(solver_name='highs')  # Usar el solver 'highs'

# Exportar rutas a un archivo CSV
vr_model.export_routes('routes.csv')

# Visualizar rutas en un gráfico
vr_model.plot_routes()

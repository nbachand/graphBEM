# General Ideas

## Building Graph
The general idea underlying this BEM scheme is to represent the building as a graph. All geometric relationships are expressed in this graph independently of any input geometry file (e.g., CAD). This structure gives the solver a lot of flexibility and removes the possibly cumbersome geometry creation step. However, the solver allows for graphs that represent unrealistic geometries and does not verify the integrity of the geometry.
### Nodes
The graph contains two type of nodes: solved nodes (e.g, rooms) and boundary nodes (e.g., the outdoors, sun/sky, floor bottom-boundary condition).

## Edges
Edges represent surfaces (e.g., walls, roofs, floors). The edge weights represent duplicate walls that can be solved as one. For instance, for a corner room with two equivalent exterior walls, I specify an edge with the properties of one exterior wall and assign a weight of 2. 
The layout is specified row-first with rooms specified along rows and columns and boundary conditions (e.g. the outside) specified only in columns. This matrix structure differentiates solved and boundary nodes, resulting in a low-rank graph matrix with more columns than rows. 

## Energy Pathways
![image](https://github.com/nbachand/graphBEM/assets/42705584/9abbd687-ef17-4445-b11f-029d940828a4)

*Note: Ventilation is not fully developed and I am not including it in my building*
# Verification

## Room Modeling

### Governing Equation
Room are solved as 0-D thermal masses with the governing equation:

$$
\rho V C_{p}\frac{dT_{int}}{dt} = E_{f} + E_{int} + E_{vt} 
$$

where $\rho$, $V$ and $C_p$ are the air density, volume, and and specific heat, respectively. Currently, the model assumes $\rho = 1.225$ and $C_{p}= 1005$. $E_f$, $E_{int}$, and $E_{vt}$ represent heat flux into and out of the room via the fabric, internal heat gains, and ventilation, respectively.

#### Discretization
The room is solved using a first order discretization in time:

$$
\rho V C_{p}\frac{T_{int}^{t+1} - T_{int}^{t}}{\Delta t} = E_{f}^{t} + E_{int}^{t} + E_{vt}^{t} 
$$

which gives:

$$
T_{int}^{t+1} = T_{int}^{t} + \frac{\Delta t}{\rho V C_{p}} (E_{f}^{t} + E_{int}^{t} + E_{vt}^{t}) 
$$

## Temperature Boundary Conditions

Temperature boundary conditions are also described with nodes. Boundary conditions do not solve for temperature, and instead take a temperature time series as input. Currently the scheme checks for the names of boundary nodes ('OD for outdoors and 'FL' for floor) and then uses the associated temperature time series instead of solving for temperature. If single temperature is given instead of a series, that temperature is used at all timesteps.

All temperature boundary conditions technically interact with the adjacent surfaces through convection. However, adjusting the convection coefficient can change the behavior of the boundary condition. For instance, a very large convection coefficient effectively sets a constant temperature at the adjacent surfaces. I use this strategy to set an (effectively) constant boundary condition for the ground.

### Outdoor Temperature
The outdoor temperature can be set with any time series. I am using energy plus data because of the corresponding radiation data.

### Floor Temperature
There are many ways to choose floor temperatures: 

## Temperature Boundary Conditions

Temperature boundary conditions are also described with nodes. Boundary conditions do not solve for temperature, and instead take a temperature time series as input. Currently the scheme checks for the names of boundary nodes ('OD for outdoors and 'FL' for floor) and then uses the associated temperature time series instead of solving for temperature. If single temperature is given instead of a series, that temperature is used at all timesteps.

All temperature boundary conditions technically interact with the adjacent surfaces through convection. However, adjusting the convection coefficient can change the behavior of the boundary condition. For instance, a very large convection coefficient effectively sets a constant temperature at the adjacent surfaces. I use this strategy to set an (effectively) constant boundary condition for the ground.

### Outdoor Temperature
The outdoor temperature can be set with any time series. I am using energy plus data because of the corresponding radiation data.

### Floor Temperature
There are many ways to choose floor temperatures: 

![image](https://github.com/nbachand/graphBEM/assets/42705584/36239624-7c2a-4b4b-b5bf-7b774556ec47)
([pdf](zotero://open-pdf/library/items/YBDPDVY7?page=5&annotation=JBJ9IRAR))  ([Gutiérrez González et al., 2022, p. 5](zotero://select/library/items/X6B88W33))

I have been using a simplified temperature model where I take the average outdoor temperature during a given time period and subtract 0.5 to 3 C. González et al. found that taking the monthly average indoor temperature and subtracting 1.5 C gave some of the best results. My method borrows from this, but avoids iteratively calculating the internal temperatures. There may be a better way...


## Wall Modeling

### Governing Equation

The wall solves one-dimensional heat conduction with spatially varying properties:

$$
\rho c_p \frac{\partial T}{\partial t}
= \frac{\partial}{\partial x}\left(k\frac{\partial T}{\partial x}\right)
$$

Here, $\rho$ is density, $c_p$ is specific heat and $k$ is conductivity.

#### Discretization

The solver uses finite volumes aligned with material layers. `n` sets a target cell width from the total solid thickness. Each solid layer has at least one cell, so `wall.n` is the resulting cell count. Layer thicknesses, conductivities and heat capacities remain unchanged. `wall.x` contains the two surface positions and the cell centers; spacing is not generally uniform.

Adjacent cells share a conductance per unit area:

$$
G_{i,i+1} = \left(\frac{\Delta x_i}{2k_i} + R_{\mathrm{gap}} + \frac{\Delta x_{i+1}}{2k_{i+1}}\right)^{-1}
$$

The gap resistance is zero where the solid layers touch. Using the same conductance in both cells conserves heat across each interface. The implicit update is:

$$
(\rho c_p\Delta x)_i\frac{T_i^{t+1}-T_i^t}{\Delta t}
=G_{i-1,i}(T_{i-1}^{t+1}-T_i^{t+1})+G_{i,i+1}(T_{i+1}^{t+1}-T_i^{t+1})
$$

The first and last cells use the surface balances below. With `implicit=False`, fluxes are evaluated at the start of the step. The explicit solver rejects timesteps that exceed its stability bound, including after a change in wind speed. It does not increase material density to stabilize a run.

### Surface Energy Balance

Each massless surface balances conduction, convection and applied radiation:

$$
G_s(T_1-T_s)+h(T_{\mathrm{air}}-T_s)+q_{\mathrm{rad}}=0
$$

Here, $G_s$ includes the adjacent half-cell resistance and any air gap between that cell and the surface. Radiation is in W/m² and positive into the surface. The surface temperature is:

$$
T_s=\frac{G_sT_1+hT_{\mathrm{air}}+q_{\mathrm{rad}}}{G_s+h}
$$

The wind-adjusted coefficient is recalculated each step and used in both the wall solve and the convective power returned to the adjoining air. The building's radiation calculation and damping supply the applied radiation.

### Construction

The wall construction is specified as a pandas DataFrame where each index corresponds to a material. Materials are ordered (from top to bottom of the DataFrame) along an edge from node 1 to node 2. This almost always means the material corresponding to the side of the wall facing indoors should be specified first.

Prior to constructing the model, I remove all materials with conductivity greater than 10 $W/(m.K)$. These are generally thin metal panelings that transfer heat an order of magnitude faster than other materials, and do not provide significant thermal storage. These high conductivity and low thermal mass materials would require a much smaller timestep to resolve.

#### Materials

Each material is specified with the following properties:
1) Thickness $[m]$
2) Conductivity $[W/(m.K)]$
3) Density $[kg/m^3]$
4) Specific Heat $[J/(kg.K)]$
5) Thermal Resistance $[m^2.K/W]$

`Material:AirGap` layers use their specified resistance without storage cells or artificial heat capacity. If no physical thickness is specified, the gap has zero plotting width. At least one solid layer is required.

### Surface Properties
Each wall has the convection coefficient $h$ and radiative absorptivity $\alpha$ specified at both the front and back surfaces.

### Solver checks

Run the physical regression checks in the project Python environment:

```sh
python -m unittest discover -s tests -v
```

These cover steady heat flow, transient storage, surface balances, changing wind, air gaps, ground boundaries and convergence toward the analytical homogeneous-slab solution.

`scripts/compare_wall_solver.py` reproduces the short Burbank diagnostic using an EnergyPlus case directory containing `surface_data.csv` and `results/eplusout.sql`. Pass `--baseline-wall` with an older `WallSimulation.py` to compare solvers. Results are saved in `analysis/energyplus_wall_fix`. This comparison uses the original 5,000 weather samples at 30-second spacing and is not a validation with matched warm-up or boundary conditions.

`scripts/verify_physics_fixes.py --ep-case /path/to/Burbank/case` repeats the EnergyPlus comparison at 30, 15 and 7.5 seconds. It checks total wall and room-air storage against external heat input, with initial and ground temperatures held fixed across grids. Results and stage-by-stage comparisons are in `analysis/physics_fixes`. `scripts/audit_physics.py` reruns the original counterexamples against the current model.

### Matched EnergyPlus boundary replay

`scripts/compare_ep_replay.py` tests the wall solver against the existing Burbank EnergyPlus SQL results. It imports the exact materials and layer order, resolves paired interzone surfaces, and treats every exterior orientation separately. The floor uses EnergyPlus's recorded 18°C ground face, reversed layer order relative to the example building, and no added soil. Surface areas weight the errors. This is a controlled conduction test; room-air temperatures, convection coefficients and radiation are not independently validated.

Two tests separate the boundary calculations from conduction:

- `robin_linear` supplies EnergyPlus air temperatures, convection coefficients and net radiative fluxes. Surface temperatures and conduction are predictions.
- `dirichlet` supplies EnergyPlus surface temperatures on both faces. Only conduction is a prediction.

The SQL contains both hourly and 15-minute time records. Only zone-timestep records enter the replay. Their values are treated as endpoint samples and interpolated between endpoints. The optional `robin` mode instead holds each interval's inputs constant, exposing sensitivity to this assumption. Predictions are sampled at the corresponding endpoints. Both faces use conduction positive from the material toward the face, following the [EnergyPlus output convention](https://bigladdersoftware.com/epx/docs/22-2/input-output-reference/group-thermal-zone-description-geometry.html).

The full August history is replayed after first-day periodic spin-up to a cell-temperature change below 0.00001 K. Metrics exclude August 1–7 because EnergyPlus's actual warm-up states are unavailable. The matched run uses a 60-second timestep and a target of 36 solid cells. It covers 21 physical surfaces, without double-counting interzone partitions.

| Construction | Inside conduction RMSE | Outside conduction RMSE |
| --- | ---: | ---: |
| Exterior wall | 0.0146 W/m² | 0.0484 W/m² |
| Roof | 0.0073 W/m² | 0.0790 W/m² |
| Floor | 0.5008 W/m² | 0.0023 W/m² |
| Partition | 0.0023 W/m² | 0.0019 W/m² |

These are the prescribed-surface-temperature results. With prescribed air temperatures, coefficients and radiation, wall surface-temperature RMSE is 0.022 K inside and 0.097 K outside. Roof values are 0.020 K and 0.070 K. The results support the repaired wall solver, while leaving the floor's smaller transient discrepancy unresolved. They do not establish agreement of the free-running buildings or explain each earlier mismatch individually.

A 30-second, 72-cell refinement on one surface of each type reduces wall and roof outside-face conduction errors to 0.006 and 0.013 W/m². Floor inside conduction changes by only 0.011 W/m² RMS between refinements. Its full-run error also remains near 0.50 W/m² when excluding two or three weeks, so it is not explained by the tested refinement or initialization exclusions. The stored hold-input runs show why temporal forcing assumptions must accompany the results.

Run with the graphBEM Python environment:

```sh
python scripts/compare_ep_replay.py --ep-case /path/to/Burbank/case --output analysis/energyplus_replay/matched
python scripts/compare_ep_replay.py --ep-case /path/to/Burbank/case --dt 30 --cells 72 --surface-ids 1 3 5 6 --output analysis/energyplus_replay/convergence
python scripts/compare_ep_replay.py --ep-case /path/to/Burbank/case --dt 900 --cells 9 --modes robin dirichlet --output analysis/energyplus_replay/coarse
python scripts/compare_ep_replay.py --ep-case /path/to/Burbank/case --dt 60 --cells 36 --modes robin dirichlet --output analysis/energyplus_replay/refined
MPLBACKEND=Agg python scripts/summarize_ep_replay.py
```

Results: [plot](analysis/energyplus_replay/comparison.png), [matched metrics](analysis/energyplus_replay/matched/metrics.csv), [refinement changes](analysis/energyplus_replay/refinement_changes.csv), and [initialization sensitivity](analysis/energyplus_replay/initialization_sensitivity.csv). Metadata records source hashes and spin-up convergence. Full compressed histories are retained locally but excluded from Git.

## Ventilation
The production example keeps ventilation disabled. The optional HWP4 path accepts scalar or vector times and temperatures, and returns volumetric flow with heat transfer positive into the room. Its existing schedule opens windows before 07:00 and after 19:00.

The top-pivoted window width follows Eq. 4 of [Hult, Iaccarino and Fischer (2012)](https://publications.ibpsa.org/proceedings/simbuild/2012/papers/simbuild2012_05b_3_Hult.pdf). The complete side-plus-bottom opening width is squared. Effective-area integration splits at the geometry breakpoint to preserve the closed-window limit.

## Radiation

Radiation is represented analogously to an electrical circuit. 

### Single surfaces

Each radiating object $i$ has an emmisive power $E_i = E_{bi} \epsilon_i$,  where $E_{bi}$ is the black body emmisive power and $\epsilon_i$ is the surface emmisivity. The surfaces are assumed to be opaque, diffuse, and grey such that the absorptivity $\alpha_i = \epsilon_i$. The radiosity $J_i$ of an object $i$ represents the sum of emmisive and reflective power. In a circuit context, the net radiation leaving a surface is calculated as the current between two nodes with potentials $E_{bi}$ and $J_i$. The equivalent resistance between  $E_{bi}$ and $J_i$ is $\frac{1 - \epsilon_i}{A_i \epsilon_i}$, where $A_i$ is the surface area.

[image] ([pdf](zotero://open-pdf/library/items/A87CE92Z?page=893&annotation=8YBYK8QF))  
([“Fundamentals of heat and mass transfer”, 2007, p. 823](zotero://select/library/items/UQZVVFCE))
### Between Objects

The effective resistance between surfaces $i$ and $j$ depends on the surface areas $A_i$ and $A_j$ and the view factor $F_{ij}$. $F_{ij}$ represents the portion of radiation leaving surface $i$ that reaches surface $j$, and depends on the orientation of the surfaces and the relative areas. $F_{ij} \neq F_{ji}$, and instead $A_{i} F_{ij} = A_{j} F_{ji}$. A larger $A_{i} F_{ij}$  means that more radiation will be transferred from surface $i$ to surface $j$. Similarly, the effective resistance between two nodes with potentials $J_i$ and $J_j$ is $(A_{i} F_{ij})^{-1}$ 

[image] ([pdf](zotero://open-pdf/library/items/A87CE92Z?page=895&annotation=9K2996R8))  
([“Fundamentals of heat and mass transfer”, 2007, p. 825](zotero://select/library/items/UQZVVFCE))

#### View Factors
View factors $F_{ij}$ are complex geometrical relationships even for relatively simple surface orientations. Currently the implemented view factors are:
1) Aligned Parallel Rectangles [image] ([pdf](zotero://open-pdf/library/items/A87CE92Z?page=887&annotation=VXDPKVJA))  ([“Fundamentals of heat and mass transfer”, 2007, p. 817](zotero://select/library/items/UQZVVFCE))
2) Perpendicular Rectangles with a Common Edge [image] ([pdf](zotero://open-pdf/library/items/A87CE92Z?page=887&annotation=GGQ353IY))  ([“Fundamentals of heat and mass transfer”, 2007, p. 817](zotero://select/library/items/UQZVVFCE))

### Representing Object Relationships

The radiation scheme involves the most granular geometrical information because it needs to capture how different radiating objects are positioned relative to each other. Therefore, the radiation solver relies on it's own set of graphs to represent this information. In the full building graph, nodes represent rooms or boundary conditions, and edges represent walls. Within this graph, each node (room) has a radiation object with it's own associated graph. 

In this radiation graph, the nodes are radiating surfaces while the edges are pathways for radiation exchange. The weights of these edges is the effective resistance between any two surfaces with radiosities $J_i$ and $J_j$: $(A_{i} F_{ij})^{-1}$ .

While the radiation solver is fairly flexible, it currently takes a keyword argument *solveType* which can be *None* (default), *room*, or *sky*. This keyword helps build the desired associated radiation graph. However, the general solver should work for any physical radiation graph, and other implementations should allow for many possible radiation schemes. 

*solveType = None* builds an empty radiation graph representing no radiation exchange. This is appropriate for nodes representing non-radiative boundary conditions (e.g., the  ground below the floor, the outside of exterior walls).
#### Rooms
For rooms, the radiative graph captures the radiation between walls. Even though all surfaces are facing the given room, the nodes are named by the associated room or boundary condition on the other side of the wall. This naming convention simply allows walls to be distinguished from one another. For example, the radiation graph for a room *R1* with exterior walls connecting to the outdoors (node *OD* in the full building graph) would name these exterior walls *OD* in the radiation graph. This naming scheme also means that room associated with the radiation graph is not a node in the radiation graph.
##### Default Room Radiation Scheme
Currently, *solveType = room* constructs a radiation graph for a rectangular room. This radiation graph does not include radiation between walls as it is generally assumed that walls will have similar temperatures. Excluding this form of radiation greatly reduces the required geometrical information. The modeled radiation pathways are therefore between the ceiling and floor, walls and floor, and walls and ceiling. For each wall that is not the ceiling or floor, the solver initializes an edge between the wall and floor and wall and ceiling. The assigned view factor for each of these edges assumes that the wall is perpendicular too - and sharing and edge with - the floor and ground. Lastly, the scheme also adds an edge between the floor and ground, assuming they are parallel. 

#### Boundary Conditions

Currently, the only radiative boundary condition is a node representing the combined effect of the sun and sky. Therefore, I will discuss the sun/sky node, although other radiative boundary conditions could be modeled similarly. 

In the larger building graph, the sun/sky is a boundary condition represented by a node. In addition to radiating surfaces, the associated radiation graph for the sun/sky must includes a node representing the radiating boundary condition. Therefore, unlike with rooms, the node associated with the radiation graph is a node in the radiation graph.

##### Exterior radiation inputs

Exterior radiation keeps incident solar and longwave irradiance separate. Pass each exterior node's `radG` as `{"shortwave": S, "longwave": L}`. Each band can be a scalar or a time series in W/m². `L` is the sum of sky and ground irradiance after view-factor weighting. Legacy scalar/array `radG` inputs denote shortwave only.

The net inward surface flux is:

$$
q_{rad}=\alpha_{solar} S+\epsilon_{thermal}(L-\sigma T_s^4)
$$

`wall.absorptivity` is the exterior solar absorptivity. Exterior thermal emissivity remains 0.9. Weather-file sky infrared already represents the sky's emitted radiation; do not apply a second sky-emissivity multiplier. The model retains the interior grey-surface assumption, using `wall.absorptivity` for interior emissivity.

Interior radiation uses shared exchange conductances selected by surface roles, independent of dictionary order. Heat flux is calculated from graph edges, which also handles emissivity 1 without dividing by zero surface resistance. A self-loop partition retains the symmetric-face approximation: both faces receive the same per-area flux, and both contribute to the enclosure's area and energy budget.

## General Building Simulation Procedure

At each timestep, the order of solving models is:
1) Radiation
2) Walls
3) Rooms

Radiation is applied directly using the previous surface temperatures, followed by the implicit wall solve and explicit room-air update. There is no timestep-dependent temporal filter. Stability and timestep refinement are checked at 30, 15 and 7.5 seconds for the Burbank diagnostic; larger timesteps are not guaranteed stable.

# My Building (Example)
![Picture1](https://github.com/user-attachments/assets/5522a357-135d-4f6d-b77e-d0e44689033d)

All rooms are $H_{R}=3 m$ tall. Most rooms are square and $L_{R}$=4 m across. The cross ventilated room is the exception, being $L_{R}$ wide and 2 $L_{R}$ long. The floor plan of each house (shown above) is 3 $L_R$ by 2 $L_R$. The roof is flat for the purposes of solar radiation. Windows are not expolicitly modeled, but would be $H_{R}/4$ by $H_{R}/4$. The two rooms in the dual-ventilated room are connected by an open door-frame $H_{R}/4$ wide by 3 $H_{R}/4$ tall, although in the BEM this is considered as a single air volume.

## Plan2EPlus Description

```
[
  [
    {"id":0,"label":"corner_ventilation","left":"0.00","top":"0.00","width":"4.00","height":"4.00","color":""},
    {"id":1,"label":"single_sided_ventilation","left":"4.00","top":"0.00","width":"4.00","height":"4.00","color":""},
    {"id":2,"label":"dual_room_ventilation","left":"0.00","top":"4.00","width":"8.00","height":"4.00","color":""},
    {"id":3,"label":"cross_ventilation","left":"8.00","top":"0.00","width":"4.00","height":"8.00","color":""}
  ]
]
```
---

### Free-running EnergyPlus building comparison

`scripts/compare_ep_free_running.py` simulates all four rooms and 21 physical opaque surfaces through August. Room temperatures, surface temperatures and heat flows evolve freely. A streaming driver uses the production wall, room and radiation models while preserving each exterior orientation. The radiation operator is precomputed from the production network and tested against its normal solver.

The benchmark imports EnergyPlus's surface areas, orientations, zone dimensions, volumes, interzone connections and construction layers. The floor plan agrees with the example's four-room layout, but the example uses 3.00 m walls instead of 3.05 m, removes window area where EnergyPlus has no windows, and adds a partition within the dual room. These differences are removed in the benchmark. The original example configuration remains a separate case.

The floor boundary is fixed at 18°C, with the exact EnergyPlus floor construction and no extra soil layer. Solar absorptivity is 0.7 and thermal emissivity is 0.9. The exterior plywood is Smooth, giving a DOE-2 roughness multiplier of 1.11 from the [EnergyPlus material-roughness table](https://bigladdersoftware.com/epx/docs/22-2/engineering-reference/outside-surface-heat-balance.html#tarp-algorithm). This replaces the example's generic multiplier of 1.64. Interior radiation now accepts an emissivity independently of solar absorptivity and an explicit enclosure height. Legacy defaults remain available.

Exterior forcing consists of EnergyPlus's reported local outdoor air temperature, local wind and incident shortwave radiation, plus horizontal infrared from the same EPW. No EnergyPlus indoor temperatures, surface temperatures, convection coefficients or net radiation enter the simulation. Using EnergyPlus's incident sunlight isolates the thermal model; it does not validate GraphBEM's solar transposition. The [EnergyPlus weather implementation](https://github.com/NREL/EnergyPlus/blob/v22.2.0/src/EnergyPlus/WeatherManager.cc) interpolates hourly infrared at timestep endpoints. The benchmark follows that timing and the first-day midnight convention. The EPW source year is mapped to the SQL run year using local standard time, without DST. An outdoor-temperature audit agrees to 7.2e-15 K.

Warm-up repeats the first day until every room and wall-cell temperature changes by less than 0.001 K between days. Both runs converged in 36 days. This matches the repeated-day procedure, but not EnergyPlus's unavailable initial state or stopping criteria: its input allows 6–25 days with a 0.4 K temperature convergence tolerance. All-month results and results excluding 7 or 14 days are saved. Early-period disagreement is larger, so the comparison should not be described as having identical warm-up states.

Refined room-temperature errors for August 8–31:

| Room | RMSE, °C | Bias, °C |
|---|---:|---:|
| Corner (zone 1) | 0.691 | +0.675 |
| Single sided (zone 2) | 0.717 | +0.706 |
| Dual room (zone 3) | 0.594 | +0.581 |
| Cross (zone 4) | 0.715 | +0.689 |

Including the full month gives room RMSE of 0.779–0.892°C. Excluding the first 14 days gives 0.543–0.673°C. These periods have different weather as well as different exposure to initialization effects.

The largest remaining flux discrepancies are exterior convection and radiation. Their errors can offset, so close room temperatures do not establish agreement in each heat-balance term. GraphBEM retains its constant interior convection coefficient, exterior convection correlation, sky/ground view factors, constant air properties and approximate enclosure radiation network, which omits wall-to-wall exchange. Split-wall view factors remain approximate even though physical areas and zone dimensions match. These algorithm differences are recorded, not tuned against EnergyPlus output.

The base run uses 60 s and 18 target wall cells; the refined run uses 30 s and 36 cells. Refinement changes room temperatures by 0.0070–0.0090 K RMS and at most 0.026 K after the first week. Exterior roof conduction changes by 0.29 W/m² RMS, so it has more numerical sensitivity than the room temperatures. The refined whole-building energy residual stays below 4.3e-07 W. Tests cover uniform equilibrium, coupled energy conservation, radiation-operator equivalence, weather timing and independence from EnergyPlus thermal predictions.

Temperatures and fluxes are compared at the 15-minute SQL endpoints. Surface errors are area weighted. Conduction is positive from the wall core toward each face; convection and radiation are positive into the face. For partitions, `outside` denotes the second room's face. Ground-side convection is a numerical constraint reaction and is excluded from flux metrics, as is the prescribed ground temperature.

```bash
python scripts/compare_ep_free_running.py --ep-case /path/to/Burbank/case --epw /path/to/Burbank.epw
python scripts/compare_ep_free_running.py --ep-case /path/to/Burbank/case --epw /path/to/Burbank.epw --dt 30 --cells 36 --output analysis/energyplus_free_running/refined
MPLBACKEND=Agg python scripts/summarize_ep_free_running.py
```

Results: [room temperatures](analysis/energyplus_free_running/refined/room_comparison.png), [interior surface heat flows](analysis/energyplus_free_running/refined/surface_comparison.png), [exterior heat flows](analysis/energyplus_free_running/exterior_comparison.png), [all metrics](analysis/energyplus_free_running/summary.csv), [refinement](analysis/energyplus_free_running/refinement.csv), and [geometry](analysis/energyplus_free_running/refined/geometry.csv). Full surface histories are retained locally and excluded from Git. Metadata records source hashes, assumptions and warm-up convergence.

### Attribution of the remaining free-running errors

`scripts/diagnose_ep_discrepancies.py` evaluates the heat-transfer laws at identical EnergyPlus temperatures, then runs independent building simulations with selected correlation changes. Production physics is not modified by these experiments. All sensitivity runs use 60 s, 18 target wall cells, converged first-day warm-up and the same geometry, materials and exterior forcing as the base benchmark. No EnergyPlus room temperatures, surface temperatures or heat flows are prescribed during the sensitivity runs.

Exterior convection is the main source of the hot roof. `convectionDOE2` averages the older wind coefficients and exponents, ignores wind direction and uses a constant natural-convection contribution of 2 W/m²K. EnergyPlus 22.2 instead selects windward or leeward coefficients suited to local surface wind and evaluates natural convection from temperature difference and tilt. At identical EnergyPlus temperatures, GraphBEM's convection flux differs by 24.9 W/m² RMS on walls and 59.1 W/m² on roofs. The average convection coefficients are 3.77 versus 4.70 W/m²K for walls and 4.10 versus 5.63 for roofs. Reconstructing the EnergyPlus calculation with the previous zone-timestep surface temperature reproduces its roof coefficient to numerical precision and wall convection to 0.0009 W/m² RMS.

The large roof radiation error mostly follows from the hotter surface. At identical surface temperatures, radiation disagreement drops from 37.2 to 1.59 W/m² RMS. Almost all of that smaller difference comes from EnergyPlus holding radiative coefficients computed from the previous 15-minute surface temperature. Reconstructing that linearization reproduces roof longwave flux to numerical precision.

The vertical-wall sky difference is distinct. GraphBEM applies the EPW horizontal infrared uniformly over the visible sky hemisphere. EnergyPlus assigns part of that exposure to air-temperature radiation: its effective sky factor is 0.3536 rather than 0.5 for a vertical wall. This approximates directional atmospheric radiation and adds about 9.8 W/m² of incident wall heat relative to GraphBEM in this case. It is a sky-model choice, not evidence of an emission-sign or emissivity error. Horizontal roofs have no such sky/air split. The small difference in Stefan–Boltzmann constants is included in the saved decomposition and is negligible here.

The controlled runs show why room-temperature agreement alone can mislead. Correcting exterior convection without changing GraphBEM's sky assumption produces a cool bias; changing the sky assumption alone increases the warm bias. The following values pool all four rooms over August 8–31:

| Diagnostic calculation | Room RMSE, °C | Room bias, °C |
|---|---:|---:|
| Base GraphBEM | 0.684 | +0.665 |
| EnergyPlus exterior convection correlation | 0.480 | −0.466 |
| EnergyPlus sky/air split | 1.201 | +1.189 |
| Both exterior changes | 0.126 | +0.001 |
| Both, plus EnergyPlus-style interior convection | 0.104 | +0.048 |

With both exterior changes, roof surface-temperature RMSE falls from 4.59 to 0.17 K. Roof convection and radiation errors fall from 38.7 and 37.1 to 0.67 and 0.42 W/m². The wall flux errors fall from about 19 and 18 to 0.55 and 0.49 W/m². These runs support the causal diagnosis without prescribing EnergyPlus thermal predictions.

Interior differences are smaller. GraphBEM's fixed h=2 exceeds EnergyPlus's mean interior coefficients of roughly 0.6–1.0 W/m²K in this case. At identical temperatures, the approximate GraphBEM enclosure radiation network differs by 1.58 W/m² RMS on exterior-wall interior faces and 1.10 on partitions, versus 0.068 on floors and 0.166 on ceilings. The omitted wall-to-wall exchange and split-wall view-factor approximations remain possible sources of this enclosure disagreement; their contributions have not been isolated from each other.

The residual is not fully attributed. The final diagnostic variant has room RMSE 0.347°C over the full month, 0.104°C after excluding one week and 0.057°C after excluding two weeks. This is consistent with a remaining initialization contribution, but the periods also have different weather. EnergyPlus's warm-up state remains unavailable. The variants retain GraphBEM's timestep coupling, constant air properties and wall discretization. Their exterior coefficients use the previous 60-second surface state, while EnergyPlus evaluates them before each 15-minute outside-surface solve. Wind direction is held at the reported 15-minute endpoint in these diagnostic runs. Whole-building energy residuals stay below 2.4e-7 W.

The practical next change is to make the exterior convection implementation consistent with the intended DOE-2 correlation and wind-height convention. Sky angular treatment should be an explicit model choice. Interior convection can then be made temperature- and orientation-dependent. The evidence does not point to another large conduction defect, though smaller conduction and initialization residuals remain.

Sources: [EnergyPlus convection implementation](https://github.com/NREL/EnergyPlus/blob/v22.2.0/src/EnergyPlus/ConvectionCoefficients.cc), [natural-convection formulas](https://github.com/NREL/EnergyPlus/blob/v22.2.0/src/EnergyPlus/ConvectionCoefficients.hh), [exterior radiation reporting](https://github.com/NREL/EnergyPlus/blob/v22.2.0/src/EnergyPlus/HeatBalanceSurfaceManager.cc), and [exterior heat-balance documentation](https://bigladdersoftware.com/epx/docs/22-2/engineering-reference/outside-surface-heat-balance.html). The source uses 7.238 in the unstable natural-convection denominator; the 22.2 Engineering Reference prints 7.283. The diagnostics follow the source and verify against the saved outputs.

```bash
python scripts/diagnose_ep_discrepancies.py --audit
python scripts/diagnose_ep_discrepancies.py --variant exterior_convection
python scripts/diagnose_ep_discrepancies.py --variant sky
python scripts/diagnose_ep_discrepancies.py --variant exterior_and_sky
python scripts/diagnose_ep_discrepancies.py --variant all_convection_and_sky
MPLBACKEND=Agg python scripts/summarize_ep_diagnosis.py
```

Results: [causal comparison plot](analysis/energyplus_diagnosis/causal_comparison.png), [sensitivity summary](analysis/energyplus_diagnosis/causal_summary.csv), [same-state exterior convection](analysis/energyplus_diagnosis/exterior_same_state.csv), [radiation decomposition](analysis/energyplus_diagnosis/radiation_decomposition.csv), and [enclosure radiation](analysis/energyplus_diagnosis/enclosure_same_state.csv). RMS components are not additive. Metadata records the audit source hashes and each variant's settings.

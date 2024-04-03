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


## Wall Modeling

### Governing Equation

The governing equation of the wall model is a 1D heat transfer equation.

$$
\rho_fC_f\frac{dT_{f}}{dt} = k_f\frac{d^2T_f}{dx^2}
$$

where $\rho_f$, $C_f$, and $k_f$ are the fabric specific heat, conductivity and density.

#### Discretization

The wall is solved using a 1-D discretization across the thickness using 9 interior nodes plus added nodes for *resistor materials* like air gaps. 

The governing equation is discretized using a forward Euler method in time and a central differencing scheme. For interior nodes, this gives:

$$
\rho_f C_f\frac{T_i^{t+1} - T_i^t}{\Delta t} = k_f \frac{T_{i-1}^t - 2 T_i^t + T_{i+1}^t}{\Delta x ^ 2}
$$

where $\rho_f$, $C_f$, and $k_f$ are the wall density, heat capacity, and conductivity, respectively. $T_i^t$ is the temperature of interior node $i$ at time $t$. Solving for $T_i^{t+1}$ gives:

$$
T_i^{t+1}  = \frac{k_f \Delta t}{\rho_f C_f \Delta x ^ 2} (T_{i-1}^t - 2 T_i^t + T_{i+1}^t) + T_i^t
$$

### Surface Energy Balance
The energy balance at the surface of the wall is

$$
E_k + E_h + E_r = 0
$$

where $E_k$, $E_h$, and $E_r$ is the energy flux towards the wall's surface from conduction, convection, and radiation. This energy balance gives:

$$
-k_fA_f\frac{dT_f}{dx} + hA_f(T_{int} - T_f) + E_r = 0
$$

where $k_f$, $A_f$, and $T_f$ are the fabric conductivity, area, and temperature, respectively. $T_{int}$ is the air temperature next to the wall. Dividing by $A_f$ and discretizing to an interior node with temperature $T_1$ at a distance $\Delta x$ inside the wall gives:

$$
k_f\frac{T_1 - T_f}{\Delta x} + h(T_{int} - T_f) + \frac{E_r}{A_f} = 0
$$

Then solving for $T_f$:

$$
T_f = \frac{T_{int} + \frac{k_f}{h\Delta x}T_1 + \frac{E_r}{hA_f}}{1 + \frac{k_f}{h\Delta x}}
$$

### Construction

The wall construction is specified as a pandas DataFrame where each index corresponds to a material. Materials are ordered (from top to bottom of the DataFrame) along an edge from node 1 to node 2. This almost always means the material corresponding to the side of the wall facing indoors should be specified first.

Prior to constructing the model, I remove all materials with conductivity greater than 10 $W/(m.K)$. These are generally thin metal panelings that transfer heat an order of magnitude faster than other materials, and do not provide significant thermal storage. These high conductivity and low thermal mass materials would require a much smaller timestep to resolve.

#### Materials

Each material is specified with the following properties:
1) Thickness $[m]$
2) Conductivity $[W/(m.K)]$
3) Density $[kg/m^3]$
4) Specific Heat $[J/(kg.K)]$
5) Thermal Resistance $[W/(m^2.K)]$
	- Thermal resistance is easier to find for some materials (e.g., air gaps) where the exact thermal properties are not listed. In this case, the model assigns the thickness to be the discretization size and assigns one node in the air gap. The conductivity is then calculated by dividing this assigned thickness by the thermal resistance. The density is set to 12 $kg/m^3$ to give a large enough timestep, and the specific heat is set to 1005  $J/(kg.K)$, the same as air. 

### Surface Properties
Each wall has the convection coefficient $h$ and radiative absorptivity $\alpha$ specified at both the front and back surfaces.

## Ventilation
There are a few specific ventilation models implemented that I used to compare with the homework, but I am currently not running ventilation in the model.

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

##### Default Sun/Sky Radiation Scheme
The sun/sky is represented as a surface with an effective radiosity $J_{sky}$. The sky is only assumed to exchange radiation with roofs, and the view factor $F_{(roof)(sky)} = 1$. This view factor assumes all radiation leaving the roof interacts with the sky, and all radiation reaching the roof comes from the sky.

The default graph for the sun/sky therefore connects all roofs to a new node representing the sun/sky.

### Solar/Sky Radiation Inputs

The current implementation avoids calculating an *effective sky temperature* $T_{sky}$ or emissive power $E_{sky}$. Instead,  $J_{sky}$ is directly inputed.  
#### Input Data
The input data (representing $J_{sky}$) is the sum of two quantities reported in energy plus weather datasets:
1) Horizontal Infrared Radiation Intensity. This represents radiation exchange with the atmosphere.
	1) “2.9.1.13 Field: Horizontal Infrared Radiation Intensity” ([“Auxiliary Programs”, p. 64](zotero://select/library/items/AK7V8DQP)) ([pdf](zotero://open-pdf/library/items/ZHREJX5S?page=64&annotation=LVWU9PH8))
	2) “IRH is defined as the rate of infrared radiation emitted from the sky falling on a horizontal upward-facing surface, in W/m2.” ([“EnergyPlus™ Version 9.6.0 Documentation”, p. 195](zotero://select/library/items/8W5MVM89)) ([pdf](zotero://open-pdf/library/items/PM47E5GS?page=195&annotation=K6PVMUMU))
2) Global Horizontal Radiation. This represents solar radiation on a horizontal surface. I believe this is currently not used in energy plus because it probably uses the angle of all surfaces and calculates the incident solar radiation from the *Direct Normal Radiation*. Instead, I assume that the roof is approximately horizontal (despite angles) and avoid this complexity.
	1) “2.9.1.14 Field: Global Horizontal Radiation” ([“Auxiliary Programs”, p. 65](zotero://select/library/items/AK7V8DQP)) ([pdf](zotero://open-pdf/library/items/ZHREJX5S?page=65&annotation=PWMABJKX))
		1) “Global Horizontal Radiation in Wh/m2. (Total amount of direct and diffuse solar radiation in Wh/m2 received on a horizontal surface during the number of minutes preceding the time indicated.) It is not currently used in EnergyPlus calculations. It should have a minimum value of 0; missing value for this field is 9999.”
	2) Without this, nighttime radiation loss is unrealistically high.

Here are these quantities plotted alongside other energy plus radiation quantities:
![[Pasted image 20240318113212.png]]
## General Building Simulation Procedure

At each timestep, the order of solving models is:
1) Radiation
2) Walls
3) Rooms


---

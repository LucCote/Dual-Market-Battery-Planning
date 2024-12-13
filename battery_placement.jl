using JuMP, Gurobi, CSV, DataFrames
const GRB_ENV = Gurobi.Env()

const countries = ["DE", "CH", "BE", "NL", "AT", "FR"]

# Saturation Model Constants
const As=[11444.35979,12239.89379,11367.75019,11305.41721,11814.47702,11904.19937]
const ks=[0.12549,0.12456,0.12372,0.12413,0.12558,0.12579]
const x0s=[19.93437,20.37391,19.92378,19.86986,20.12401,20.12008]

# Linear Model Constants
const as=[3.47004,24.29605,24.18950,7.70488,22.11663,7.49478] # intraday
# const as=[3.41949,28.97329,22.72094,7.64332,22.70639,8.24277]  # no intraday

const bs=[-309.16140,-44.89129,-141.27930,-28.16586,-53.17825,-161.83537] # intraday
# const bs=[-260.61280,-58.06762,-73.18494,29.70322,-29.35486,-157.17051]  # no intraday

function profit_model(x, A, k, x0)
  return A*((exp(-k*(x-x0))*k)/(1 + exp(-k*(x-x0))^2))
end

function wind_solar_model(x,a,b)
   return a*x+b
end

# Generate deployment matrix for the EU-wide doubling case
function make_doubling_case()
  VREs = [151.2,5.93,14.05,34.65,10.81,42.75]
  start_offset = 6 # starting month of july
  T=60
  C=6
  deployment = zeros((C,Int(ceil((T+start_offset)/12))))
  rate = 2/(6*12)
  for c=1:C
    for t=1:T
      year = Int(ceil((t+start_offset)/12))
      deployment[c,year] = VREs[c]*(1+rate*(t+6))
    end
  end
  return deployment
end

# Build single stage model
function build_model(scenarios, prob)
    delta = 0.8
    T=60
    C=6
    B=zeros(T)
    B[1] = 1
    B[Int(ceil(T/2))] = 0
    omega = size(scenarios,1)
    println("omega", omega)

    # generate profit matrix for each country, scenario, and timestep
    P=zeros((omega,T,C))
    for s = 1:omega
      for c=1:C
        for t=1:T
          start_offset = 6
          year = Int(ceil((t+start_offset)/12))
          println(countries[c], ", ",profit_model(48+t,As[c],ks[c],x0s[c]), " ............ ", wind_solar_model(scenarios[s][c,year], as[c], bs[c]))
          P[s,t,c] = profit_model(48+t,As[c],ks[c],x0s[c]) + wind_solar_model(scenarios[s][c,year], as[c], bs[c])
        end
      end
    end

    SO = Model()

    @variable(SO, i[1:T,1:C]>=0) # investment in each period
    @variable(SO, r[1:T,1:C]>=0) # total resources in each period
    @variable(SO, cumprof[1:T]>=0) # total profit in each period
    @variable(SO, wc[1:C]>=0) # worst case profit for each country

    # Robust/Risk constraints
    @constraint(SO, [s=1:omega, c=1:C], wc[c] <= sum(P[s,t,c]*r[t,c] for t=1:T))
    @constraint(SO, sum(wc[c] for c=1:C) >= delta*sum(prob[c,s]*P[s,t,c]*r[t,c] for t=1:T,c=1:C,s=1:omega))
    
    # Budget constraint
    @constraint(SO, [t=1:T], sum(i[t,c] for c=1:C) <= B[t])

    # Deployed resources
    @constraint(SO, [t=1:T,c=1:C], r[t,c] == sum(i[t2,c] for t2=1:t))

    # Calculate total profit
    @constraint(SO, [t2=1:T], cumprof[t2] == sum(prob[c,s]*P[s,t,c]*r[t,c] for t=1:t2,c=1:C,s=1:omega))

    @objective(SO, Max, sum(prob[c,s]*P[s,t,c]*r[t,c] for t=1:T,c=1:C,s=1:omega))

    return SO
end

function build_twostage_model(scenarios, prob)
    delta = 0.8
    T=60
    C=6
    B = 1
    T2 = Int(ceil(T/2))
    W = size(scenarios,1) # number of country specific scenarios
    WR = W^C # number of general scenarios
    MC = 1000*ones(C,C) # cost of relocating 100kWh between countries

    # generate profit matrix for each country, scenario, and timestep
    P=zeros((W,T,C))
    for s = 1:W
      for c=1:C
        for t=1:T
          start_offset = 6
          year = Int(ceil((t+start_offset)/12))
          println(countries[c], ", ",profit_model(48+t,As[c],ks[c],x0s[c]), " ............ ", wind_solar_model(scenarios[s][c,year], as[c], bs[c]))
          P[s,t,c] = profit_model(48+t,As[c],ks[c],x0s[c]) + wind_solar_model(scenarios[s][c,year], as[c], bs[c])
        end
      end
    end

    # generate general scenarios and corresponding probabilities
    prob2 = zeros(WR)
    CS = zeros(Int,WR,C)
    for s=1:WR
      S = zeros(Int,C)
      for c=1:C
        S[c] = mod(floor(s/(W^(c-1))),W)+1
        CS[s,c] = S[c]
      end
      prob2[s] = prod(prob[c,S[c]] for c=1:C)
    end

    SO = Model()

    @variable(SO, i[1:C]>=0)
    @variable(SO, r[1:WR,1:C]>=0)
    @variable(SO, x[1:WR,1:C,1:C]>=0)
    @variable(SO, p[1:WR]>=0) # profit of each scenario
    @variable(SO, cumprof[1:T]>=0)
    
    # Robust/risk constraint
    @constraint(SO, [s=1:WR], p[s] == sum(P[CS[s,c],t,c]*i[c] for t=1:T2-1,c=1:C) + sum((P[CS[s,c],t,c]*r[s,c]) for t=T2:T,c=1:C) - sum((MC[c1,c2]*x[s,c1,c2]) for c1=1:C,c2=1:C))
    @constraint(SO, [s=1:WR], p[s] >= delta*(sum(prob2[s2]*(P[CS[s2,c],t,c]*i[c]) for t=1:T2-1,c=1:C,s2=1:WR) + sum(prob2[s2]*(P[CS[s2,c],t,c]*r[s2,c]) for t=T2:T,c=1:C,s2=1:WR) - sum(prob2[s2]*(MC[c1,c2]*x[s2,c1,c2]) for c1=1:C,c2=1:C,s2=1:WR)))
 
    # Budget constraint
    @constraint(SO, sum(i[c] for c=1:C) <= B)

    # Link country resources and relocations
    @constraint(SO, [s=1:WR,c=1:C], r[s,c] == i[c]+sum(x[s,c2,c]-x[s,c,c2] for c2=1:C))

    @constraint(SO, [t2=1:T], cumprof[t2] == sum(prob2[s]*(P[CS[s,c],t,c]*i[c]) for t=1:min(T2-1,t2),c=1:C,s=1:WR) + sum(prob2[s]*(P[CS[s,c],t,c]*r[s,c]) for t=T2:min(T,t2),c=1:C,s=1:WR) - ((t2 >= T2) ? 1 : 0)*sum(prob2[s]*(MC[c1,c2]*x[s,c1,c2]) for c1=1:C,c2=1:C,s=1:WR))

    @objective(SO, Max, sum(prob2[s]*(P[CS[s,c],t,c]*i[c]) for t=1:T2-1,c=1:C,s=1:WR) + sum(prob2[s]*(P[CS[s,c],t,c]*r[s,c]) for t=T2:T,c=1:C,s=1:WR) - sum(prob2[s]*(MC[c1,c2]*x[s,c1,c2]) for c1=1:C,c2=1:C,s=1:WR))

    return SO
end

# Run single stage case
function run_model()
  main_case = Matrix(CSV.read("main_case.csv",DataFrame,header=0))
  accel_case = Matrix(CSV.read("accelerated_case.csv",DataFrame,header=0))
  doubling_case = make_doubling_case()
  probs = 0.25*ones((length(countries),3))
  for i=1:length(countries)
    probs[i,2] = 0.5
  end
  SO = build_model([main_case, accel_case, doubling_case],probs)
  set_optimizer(SO, optimizer_with_attributes(Gurobi.Optimizer,"LogToConsole" => 1))
  optimize!(SO)
  return SO
end

SO = run_model()

i = value.(SO[:i])
r = value.(SO[:r])
for c=1:6
  println(countries[c]," r: ", r[:,c])
end

println("Expected profits: ", value.(SO[:cumprof]))

# Run two stage case
function run_model_twostage()
  main_case = Matrix(CSV.read("main_case.csv",DataFrame,header=0))
  accel_case = Matrix(CSV.read("accelerated_case.csv",DataFrame,header=0))
  doubling_case = make_doubling_case()
  probs = 0.25*ones((length(countries),3))
  for i=1:length(countries)
    probs[i,2] = 0.5
  end
  SO = build_twostage_model([main_case, accel_case, doubling_case],probs)
  set_optimizer(SO, optimizer_with_attributes(Gurobi.Optimizer,"LogToConsole" => 1))
  optimize!(SO)
  return SO
end

SO = run_model_twostage()

i = value.(SO[:i])
r = value.(SO[:r])
for s=1:3^6
  for c=1:6
    println(countries[c], "s=",s , "i:",i[c], "r: ", r[s,c])
  end
end

println("Expected profits: ", value.(SO[:cumprof]))
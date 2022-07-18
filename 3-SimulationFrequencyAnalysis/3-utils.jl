"""
    load_discharge_simulations(filename::String)

Load the discharge simulations from the NetCDF file `filename`.
"""
function load_discharge_simulations(filename::String)

    scenario_id = string.(ncread(filename, "scenario_id"))
    data = ncread(filename, "Dis")

    @assert ncgetatt(filename, "time", "units") == "days since 1950-01-01 00:00:00"
    years = year.(Date(1950, 1, 1) .+ Day.(ncread(filename, "time")))
    
    output = DataFrame(Ensemble = String[],
              GCM = String[],
              RCM = String[],
              Member = String[],
              Scenario = String[],
              Years = Array{Int64}[],
              Discharges = Array{Float64}[])

    for index in 1:size(data,2)

        ensemble = string(scenario_id[34:36, index]...)  
        gcm = string(scenario_id[38:40, index]...)  
        rcm = string(scenario_id[42:44, index]...)  
        member = string(scenario_id[46:48, index]...) 
        scenario = get_scenario(member)
        
        dis = DataFrame(Year = years, Discharge = data[:, index])
        allowmissing!(dis)
        replace!(dis.Discharge, 1.0e20 => missing)
        dropmissing!(dis)
        
        push!(output, [ensemble, gcm, rcm, member, scenario, dis.Year, dis.Discharge])
    end
    
    return output
end

function get_scenario(member::String)
    if occursin(r"R4.", member) || occursin(r"4..", member)
        scenario = "RCP45"
    elseif occursin(r"R8.", member) || occursin(r"8..", member)
        scenario = "RCP85"
    elseif occursin(r"K..", member)
        scenario = "RCP85"
    else
        scenario = missing
    end
    return scenario
end
function getyears(fm::PseudoMaximaEVA)
    return getproperty.(fm.model.datadistribution, :year)
end

function postprocess(fm_obs::PseudoMaximaEVA, fm_sim::HierarchicalBayesEVA, ref_year::Int64, model::Int64)
        
    @assert model <= length(fm_sim.fittedmodels)
    years_obs = getyears(fm_obs)[1];
    years_sim = collect(1955:1:2099);  ## !!à modifier selon les données!!
    
    @assert ref_year in years_obs
    idx_ref_obs = findall(==(ref_year), years_obs)[1]
    idx_ref_sim = findall(==(ref_year), years_sim)[1]

    # GEV PARAMETERS
    if ErrorsInVariablesExtremes.isstationary(fm_obs.model)
        obspd = ErrorsInVariablesExtremes.getdistribution(fm_obs)
    else
        obspd = ErrorsInVariablesExtremes.getdistribution(fm_obs)[:,idx_ref_obs]
    end
    simpd = Extremes.getdistribution(fm_sim.fittedmodels[model])

    # OBS (REF)
    μ₁ = Distributions.location.(obspd)
    σ₁ = Distributions.scale.(obspd)
    ξ = Distributions.shape.(obspd)

    # IF FM_SIM IS STATIONNARY
    if size(simpd, 2)==1
        μ₁ = μ₁ .* ones(length(years_sim))'
        return GeneralizedExtremeValue.(μ₁, σ₁, ξ)
    end

    # SIM (REF)
    μ₂ = Distributions.location.(simpd[:, idx_ref_sim])  
    σ₂ = Distributions.scale.(simpd[:, idx_ref_sim])

    # SIM (FUT)
    μ₃ = Distributions.location.(simpd)
    σ₃ = Distributions.scale.(simpd)

    # CORRECTION 
    μ = μ₃ .+ σ₃ ./ σ₂ .* (μ₁ .- μ₂);
    σ = σ₁ .* σ₃ ./ σ₂ 

    # SIM (FUT) BIAS CORRECTED    
    return GeneralizedExtremeValue.(μ, σ, ξ)
end

function postprocess(fm_obs::PseudoMaximaEVA, fm_sim::HierarchicalBayesEVA, ref_year::Int64)
    
    n = length(fm_sim.fittedmodels)
    
    pm₁ = postprocess(fm_obs, fm_sim, ref_year, 1)
    
    iterations, years = size(pm₁)
    
    output = Array{GeneralizedExtremeValue{Float64}, 3}(undef, (iterations, years, n))
    
    output[:,:,1] = pm₁
    
    @showprogress for i = 2:n
        output[:,:,i] = postprocess(fm_obs, fm_sim, ref_year, i)
    end
    
    return output
end

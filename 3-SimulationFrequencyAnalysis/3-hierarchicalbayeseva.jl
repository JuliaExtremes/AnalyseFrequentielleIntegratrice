struct HierarchicalBayesEVA{T} <: fittedEVA{T}
    model::HierarchicalBayesModel
    hyperparameters::Mamba.Chains
    fittedmodels::Vector{Extremes.BayesianEVA{T}}
end

struct HierarchicalBayesEVAStd{T} <: fittedEVA{T}
    model::HierarchicalBayesModel
    hyperparameters::Mamba.Chains
    fittedmodels::Vector{Extremes.BayesianEVA{T}}
    scale::Float64
    offset::Float64
end

import Extremes.showfittedEVA

function showfittedEVA(io::IO, obj::HierarchicalBayesEVA; prefix::String = "")
    println(io, prefix, "HierarchicalBayesEVA")
    println(io, prefix, "   model: ", typeof(obj.model))
    println(io, prefix, "   hyperparameters :")
    Extremes.showChain(io, obj.hyperparameters, prefix = "\t\t")
    println(io, prefix, "   fittedmodels :\t\t", typeof(obj.fittedmodels), "[", length(obj.fittedmodels), "]")
    println(io)
end

function showfittedEVA(io::IO, obj::HierarchicalBayesEVAStd; prefix::String = "")
    println(io, prefix, "HierarchicalBayesStdEVA")
    println(io, prefix, "   model: ", typeof(obj.model))
    println(io, prefix, "   hyperparameters :")
    Extremes.showChain(io, obj.hyperparameters, prefix = "\t\t")
    println(io, prefix, "   fittedmodels :\t\t", typeof(obj.fittedmodels), "[", length(obj.fittedmodels), "]")
    println(io, prefix, "   scale :\t\t", typeof(obj.scale))
    println(io, prefix, "   offset :\t\t", typeof(obj.offset))
    println(io)
end

function invgamma_sampling(y::AbstractVector{<:Real}, α::Real=0.01, β::Real=0.01)

    n = length(y)
    s² = (n-1) * var(y)
    m = mean(y)

    pd = InverseGamma(n/2+α, s²/2 + β)
    σ² = rand(pd)
    σ = sqrt(σ²)

    std = σ/sqrt(n)
    pd = Normal(m, std)
    μ = rand(pd)

    return μ, σ
end

function fitbayes(model::HierarchicalBayesModel; 
    δ₀::Real=0.5,
    warmup::Int=10000,
    thin::Int=10,
    niter::Int=20000,
    adapt::Symbol=:warmup)
    
    # Hyperprior check
    @assert typeof(model.hyperprior[1]) == Flat
    @assert typeof(model.hyperprior[2]) == InverseGamma{Float64}
    α, β = Distributions.params(model.hyperprior[2])
    
    #standardization
    data_layer_std, offset, scale = standardize_df(model.data);

    m = length(data_layer_std)
    n = Extremes.nparameter(data_layer_std[1])

    #vector initialization
    params = zeros(m, n, niter)
    ν = zeros(n, niter)
    τ = ones(n, niter)

    #acceptance counts for the Metropolis-Hastings step
    acc = zeros(m)

    #initialization
    Σ = Matrix{Float64}[]

    for i in 1:m
        fd = Extremes.fit(data_layer_std[i])
        params[i, :, 1] = fd.θ̂
        push!(Σ, inv(Symmetric(Extremes.hessian(fd))))
    end
    for i in 1:m
        if !isposdef(Σ[i])
            Σ[i] = Matrix(I, n, n)
        end
    end
    δ = δ₀ * ones(m)
    ν[:,1] = mean(params[:,:,1], dims=1)
    τ[:,1] = std(params[:,:,1], dims=1)
    u = rand(m)


    @showprogress for iter=2:niter

        #Updating the data layer parameters

        rand!(u)

        for i = 1:m

            #Normal random walk is used for Metropolis-Hastings step
            candidates = rand( MvNormal( params[i , : , iter-1], δ[i]*Σ[i] ) )

            logpd = Extremes.loglike(data_layer_std[i], candidates) -
                    Extremes.loglike(data_layer_std[i], params[i , : , iter-1]) +
                    logpdf(MvNormal(ν[:, iter-1], diagm(τ[:, iter-1])), candidates) -
                    logpdf(MvNormal(ν[:, iter-1], diagm(τ[:, iter-1])), params[i , : , iter-1])

            if logpd > log(u[i])
                params[i , : , iter] = candidates
                acc[i] += 1
            else
                params[i , : , iter] = params[i , : , iter-1]
            end
        end

        # Updating the process layer parameters
        for j in 1:n
            ν[j,iter], τ[j,iter] = invgamma_sampling(params[:, j, iter], α, β)
        end

        # Adapting the stepsize
        if iter % 50 == 0
            if !(adapt == :none)
                if (iter <= warmup) | (adapt==:all)
                    accrate = acc ./ 50
                    δ = ErrorsInVariablesExtremes.update_stepsize.(δ, accrate)
                    acc = zeros(m)
                    for i in 1:m
                        covMat = StatsBase.cov(params[i, :, 1:iter]')
                        Σ[i] =  covMat .+ 1e-4 * tr(covMat) * Matrix(I, n, n)
                    end
                end
            end
        end
    end

    #Extracting output
    parmnames = String[]
    res = Array{Float64, 2}

    if n == 3
        parmnames = ["ν_μ", "ν_ϕ", "ν_ξ", "τ_μ", "τ_ϕ", "τ_ξ"]
        res = vcat(ν, τ)
        
        fm = [Extremes.BayesianEVA(data_layer_std[i], Mamba.Chains(collect(params[i,:,:]')[warmup:thin:niter, :], names=["μ", "ϕ", "ξ"])) for i=1:m]
    end

    if n == 4
        parmnames = ["ν_μ₀", "ν_μ₁", "ν_ϕ", "ν_ξ", "τ_μ₀", "τ_μ₁", "τ_ϕ", "τ_ξ"]
        res = vcat(ν, τ)
        
        fm = [Extremes.BayesianEVA(data_layer_std[i], Mamba.Chains(collect(params[i,:,:]')[warmup:thin:niter, :], names=["μ₀", "μ₁", "ϕ", "ξ"])) for i=1:m]
    end

    if n == 5
        parmnames = ["ν_μ₀", "ν_μ₁", "ν_ϕ₀", "ν_ϕ₁", "ν_ξ", "τ_μ₀", "τ_μ₁", "τ_ϕ₀", "τ_ϕ₁", "τ_ξ"]
        res = vcat(ν, τ)
        
        fm = [Extremes.BayesianEVA(data_layer_std[i], Mamba.Chains(collect(params[i,:,:]')[warmup:thin:niter, :], names=["μ₀", "μ₁", "ϕ₀", "ϕ₁", "ξ"])) for i=1:m]
    end
    
    res = Mamba.Chains(collect(res'), names=parmnames)
    res = res[warmup:thin:niter, :, :]
    
    # TO-DO: print acceptance rate / exploration stepsize
    #println( "Exploration stepsize after warmup: ", δ )
    #println( "Acceptance rate: ", mean(acc) / (niter-warmup) )
    
    println("?? parameters acceptance rate")
    #println("    ", vec(mean(acc[:, warmup:niter], dims=2)))
    println("")

    println("?? mean acceptance rate")
    #println("    ", mean(acc_y[:, warmup:niter]))
    println("")
    
    return HierarchicalBayesEVAStd(model, res, fm, scale, offset)
end

function standardize_values(values::Vector{Vector{Float64}})
    
    offset = mean(vcat(values...))
    scale = std(vcat(values...))
    
    standardized_values = copy(values)
    for i in eachindex(values)
        standardized_values[i] = (values[i] .- offset) ./ scale
    end
    
    return standardized_values, offset, scale
end

function standardize_df(models::Vector{BlockMaxima})
    
    # Maxima standardization
    values = getproperty.(getproperty.(models, :data), :value)
    standardized_values, offset, scale = standardize_values(values)
    
    # Check for covariates
    locationcov = !isempty(models[1].location.covariate)
    logscalecov = !isempty(models[1].logscale.covariate)
    shapecov = !isempty(models[1].shape.covariate)
    
    # Covariate standardization
    if locationcov
        loc_names = getproperty.(first.(getproperty.(getproperty.(models, :location), :covariate)), :name)
        loc_values = getproperty.(first.(getproperty.(getproperty.(models, :location), :covariate)), :value)
        loc_standardized_values, loc_offset, loc_scale = standardize_values(loc_values)
    end
    
    if logscalecov
        logscale_names = getproperty.(first.(getproperty.(getproperty.(models, :location), :covariate)), :name)
        logscale_values = getproperty.(first.(getproperty.(getproperty.(models, :location), :covariate)), :value)
        logscale_standardized_values, logscale_offset, logscale_scale = standardize_values(logscale_values)
    end
    
    if shapecov
        shape_names = getproperty.(first.(getproperty.(getproperty.(models, :location), :covariate)), :name)
        shape_values = getproperty.(first.(getproperty.(getproperty.(models, :location), :covariate)), :value)
        shape_standardized_values, shape_offset, shape_scale = standardize_values(shape_values)
    end
        
    model_list = similar(models)

    for i in eachindex(models)
        if locationcov && logscalecov
            model_list[i] = BlockMaxima(Variable("y", standardized_values[i]),
                                    locationcov = [VariableStd(loc_names[i], loc_standardized_values[i], loc_offset, loc_scale)],
                                    logscalecov = [VariableStd(logscale_names[i], logscale_standardized_values[i], logscale_offset, logscale_scale)])
        elseif locationcov && !logscalecov
            model_list[i] = BlockMaxima(Variable("y", standardized_values[i]),
                                    locationcov = [VariableStd(loc_names[i], loc_standardized_values[i], loc_offset, loc_scale)])
        elseif !locationcov && logscalecov
            model_list[i] = BlockMaxima(Variable("y", standardized_values[i]),
                                    logscalecov = [VariableStd(logscale_names[i], logscale_standardized_values[i], logscale_offset, logscale_scale)])
        else
            model_list[i] = BlockMaxima(Variable("y", standardized_values[i]))
        end
    end
        
    return model_list, offset, scale
end

import Extremes.returnlevel   
import Extremes.cint

function returnlevel(fm::HierarchicalBayesEVA{BlockMaxima}, returnPeriod::Real)::ReturnLevel
    
    if !isempty(fm.fittedmodels[1].model.location.covariate) || !isempty(fm.fittedmodels[1].model.logscale.covariate) || !isempty(fm.fittedmodels[1].model.shape.covariate)
        Q = Matrix{Float64}(undef, 0, length(fm.fittedmodels[1].model.data.value))  # 145 années, pas sûr de l'initialisation...
    else
        Q = Matrix{Float64}(undef, 0, 1)
    end
    
    for m in fm.fittedmodels
        
        r = returnlevel(m, returnPeriod)
        
        Q = vcat(Q, r.value)
    end
    
    return Extremes.ReturnLevel(Extremes.BlockMaximaModel(fm), returnPeriod, Q) 
    
end 


function cint(rl::ReturnLevel{HierarchicalBayesEVA{BlockMaxima}}, confidencelevel::Real=.95)::Vector{Vector{Real}}
                                                                                            
      @assert rl.returnperiod > zero(rl.returnperiod) "the return period should be positive."
      @assert zero(confidencelevel)<confidencelevel<one(confidencelevel) "the confidence level should be in (0,1)."
                                                                                            
      α = (1 - confidencelevel)                                                             
                                                                                            
      Q = Chains(rl.value)                                                                  
                                                                                            
      ci = Mamba.hpd(Q, alpha = α)                                                          
                                                                                            
      return Extremes.slicematrix(ci.value[:,:,1], dims=2)                                  
                                                                                            
end 


function transform(fm::HierarchicalBayesEVAStd{BlockMaxima})
    
    scale = fm.scale
    offset = fm.offset
    
    locationcov = !isempty(fm.fittedmodels[1].model.location.covariate)
    logscalecov = !isempty(fm.fittedmodels[1].model.logscale.covariate)

    if locationcov && logscalecov
        @assert Extremes.nparameter(fm.fittedmodels[1].model) == 5
        ν_μ₀ = fm.hyperparameters.value[:,1] .* scale .+ offset;
        ν_μ₁ = fm.hyperparameters.value[:,2] .* scale;
        ν_ϕ₀ = fm.hyperparameters.value[:,3] .+ log(scale)
        ν_ϕ₁ = fm.hyperparameters.value[:,4]
        ν_ξ = fm.hyperparameters.value[:,5]
        
        τ_μ₀ = fm.hyperparameters.value[:,6] .* scale
        τ_μ₁ = fm.hyperparameters.value[:,7] .* scale
        τ_ϕ₀ = fm.hyperparameters.value[:,8]
        τ_ϕ₁ = fm.hyperparameters.value[:,9]
        τ_ξ = fm.hyperparameters.value[:,10];
        
        hparams = hcat(ν_μ₀, ν_μ₁, ν_ϕ₀, ν_ϕ₁, ν_ξ, τ_μ₀, τ_μ₁, τ_ϕ₀, τ_ϕ₁, τ_ξ)
    elseif locationcov && !logscalecov
        @assert Extremes.nparameter(fm.fittedmodels[1].model) == 4
        ν_μ₀ = fm.hyperparameters.value[:,1] .* scale .+ offset;
        ν_μ₁ = fm.hyperparameters.value[:,2] .* scale;
        ν_ϕ = fm.hyperparameters.value[:,3] .+ log(scale)
        ν_ξ = fm.hyperparameters.value[:,4]
        
        τ_μ₀ = fm.hyperparameters.value[:,5] .* scale
        τ_μ₁ = fm.hyperparameters.value[:,6] .* scale
        τ_ϕ = fm.hyperparameters.value[:,7]
        τ_ξ = fm.hyperparameters.value[:,8];
        
        hparams = hcat(ν_μ₀, ν_μ₁, ν_ϕ, ν_ξ, τ_μ₀, τ_μ₁, τ_ϕ, τ_ξ)
    elseif !locationcov && logscalecov
        @assert Extremes.nparameter(fm.fittedmodels[1].model) == 4
        ν_μ = fm.hyperparameters.value[:,1] .* scale .+ offset;
        ν_ϕ₀ = fm.hyperparameters.value[:,2] .+ log(scale)
        ν_ϕ₁ = fm.hyperparameters.value[:,3]
        ν_ξ = fm.hyperparameters.value[:,4]
        
        τ_μ = fm.hyperparameters.value[:,5] .* scale
        τ_ϕ₀ = fm.hyperparameters.value[:,6]
        τ_ϕ₁ = fm.hyperparameters.value[:,7]
        τ_ξ = fm.hyperparameters.value[:,8];
        
        hparams = hcat(ν_μ, ν_ϕ₀, ν_ϕ₁, ν_ξ, τ_μ, τ_ϕ₀, τ_ϕ₁, τ_ξ)
    else
        @assert Extremes.nparameter(fm.fittedmodels[1].model) == 3
        ν_μ = fm.hyperparameters.value[:,1] .* scale .+ offset;
        ν_ϕ = fm.hyperparameters.value[:,2] .+ log(scale)
        ν_ξ = fm.hyperparameters.value[:,3]

        τ_μ = fm.hyperparameters.value[:,4] .* scale
        τ_ϕ = fm.hyperparameters.value[:,5]
        τ_ξ = fm.hyperparameters.value[:,6];
        
        hparams = hcat(ν_μ, ν_ϕ, ν_ξ, τ_μ, τ_ϕ, τ_ξ)
    end
        
    new_hyperparameters = deepcopy(fm.hyperparameters)
    new_hyperparameters.value[:,:,1] = hparams
    
    new_fittedmodels = transform.(fm.fittedmodels, Ref(scale), Ref(offset))
    
    return HierarchicalBayesEVA(fm.model, new_hyperparameters, new_fittedmodels)
end

function transform(fm::BayesianEVA{BlockMaxima}, scale::Float64, offset::Float64)
    
    locationcov = !isempty(fm.model.location.covariate)
    logscalecov = !isempty(fm.model.logscale.covariate)
    
    if locationcov && logscalecov
        @assert Extremes.nparameter(fm.model) == 5
        
        μ₀_std = fm.sim.value[:,1]
        μ₁_std = fm.sim.value[:,2]
        ϕ₀_std = fm.sim.value[:,3]
        ϕ₁_std = fm.sim.value[:,4]
        ξ_std = fm.sim.value[:,5];

        μ₀ = μ₀_std .* scale .+ offset;
        μ₁ = μ₁_std .* scale
        ϕ₀ = ϕ₀_std .+ log(scale)
        ϕ₁ = ϕ₁_std
        ξ = ξ_std;
        
        nparams = hcat(μ₀, μ₁, ϕ₀, ϕ₁, ξ)
        
    elseif locationcov && !logscalecov
        @assert Extremes.nparameter(fm.model) == 4
        
        μ₀_std = fm.sim.value[:,1]
        μ₁_std = fm.sim.value[:,2]
        ϕ_std = fm.sim.value[:,3]
        ξ_std = fm.sim.value[:,4];

        μ₀ = μ₀_std .* scale .+ offset;
        μ₁ = μ₁_std .* scale
        ϕ = ϕ_std .+ log(scale)
        ξ = ξ_std;
        
        nparams = hcat(μ₀, μ₁, ϕ, ξ)
        
    elseif !locationcov && logscalecov
        @assert Extremes.nparameter(fm.model) == 4
        
        μ_std = fm.sim.value[:,1]
        ϕ₀_std = fm.sim.value[:,3]
        ϕ₁_std = fm.sim.value[:,4]
        ξ_std = fm.sim.value[:,5];

        μ = μ_std .* scale .+ offset;
        ϕ₀ = ϕ₀_std .+ log(scale)
        ϕ₁ = ϕ₁_std
        ξ = ξ_std;
        
        nparams = hcat(μ, ϕ₀, ϕ₁, ξ)
        
    else
        @assert Extremes.nparameter(fm.model) == 3
        
        μ_std = fm.sim.value[:,1]
        ϕ_std = fm.sim.value[:,2]
        ξ_std = fm.sim.value[:,3];

        μ = μ_std .* scale .+ offset;
        ϕ = ϕ_std .+ log(scale)
        ξ = ξ_std;
        
        nparams = hcat(μ, ϕ, ξ)
    end
        
    new_fm = deepcopy(fm)
    new_fm.sim.value[:,:,1] = nparams 
    
    return BayesianEVA(transform(fm.model, scale, offset), new_fm.sim)
end

function transform(model::BlockMaxima, scale::Float64, offset::Float64)
    
    return BlockMaxima(Extremes.Variable(model.data.name, model.data.value .* scale .+ offset),
                model.location, model.logscale, model.shape)
end

function dic(fm::HierarchicalBayesEVAStd{BlockMaxima})
    
    data = getproperty.(fm.fittedmodels, :model)
    m = length(data)
    np = Extremes.nparameter(data[1])
    
    val = getproperty.(getproperty.(fm.fittedmodels, :sim), :value);
    mean_params = vec.(dropdims.(mean.(val, dims=1), dims=3));
    
    ν = fm.hyperparameters.value[:,1:np]
    τ = fm.hyperparameters.value[:,np+1:2*np]
    
    mean_ν = vec(mean(ν, dims=1))
    mean_τ = vec(mean(τ, dims=1));
    
    dic₁ = sum(Extremes.loglike.(data, mean_params)) + sum([logpdf(MvNormal(mean_ν, diagm(mean_τ)), params) for params in mean_params])
    dic₂ = 0.0
    
    ν₂ = Extremes.slicematrix(ν, dims = 2)
    τ₂ = Extremes.slicematrix(τ, dims = 2);
    
    for i in 1:length(data)
        e = val[i][:,:]
        params = Extremes.slicematrix(e, dims=2)
        
        d₁ = mean([Extremes.loglike(data[i], p) for p in params])
        d₂ = mean([logpdf(MvNormal(ν₂[p], diagm(vec(τ₂[p]))), params[p]) for p in 1:length(params)])
        
        dic₂ += (d₁+d₂)
    end
    
    return -2 * dic₂ + dic₁
end

function dic(fm::HierarchicalBayesEVA{BlockMaxima})
    
    data = getproperty.(fm.fittedmodels, :model)
    m = length(data)
    np = Extremes.nparameter(data[1])
    
    val = getproperty.(getproperty.(fm.fittedmodels, :sim), :value);
    mean_params = vec.(dropdims.(mean.(val, dims=1), dims=3));
    
    ν = fm.hyperparameters.value[:,1:np]
    τ = fm.hyperparameters.value[:,np+1:2*np]
    
    mean_ν = vec(mean(ν, dims=1))
    mean_τ = vec(mean(τ, dims=1));
    
    dic₁ = sum(Extremes.loglike.(data, mean_params)) + sum([logpdf(MvNormal(mean_ν, diagm(mean_τ)), params) for params in mean_params])
    dic₂ = 0.0
    
    ν₂ = Extremes.slicematrix(ν, dims = 2)
    τ₂ = Extremes.slicematrix(τ, dims = 2);
    
    for i in 1:length(data)
        e = val[i][:,:]
        params = Extremes.slicematrix(e, dims=2)
        
        d₁ = mean([Extremes.loglike(data[i], p) for p in params])
        d₂ = mean([logpdf(MvNormal(ν₂[p], diagm(vec(τ₂[p]))), params[p]) for p in 1:length(params)])
        
        dic₂ += (d₁+d₂)
    end
    
    return -2 * dic₂ + dic₁
end

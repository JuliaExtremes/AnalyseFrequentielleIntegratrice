# Fonctions utiles pour la Partie 1

"""
    get_discharge_from_txt(url::String)

Retourne un DataFrame des données journalières de débits historique à partir de l'adresse ```url``` du fichier .txt.

"""
function get_discharge_from_txt(url::String)
    
    ## TODO : Traiter les missings? + verif si années complètes ??

    # Téléchargement du fichier .txt
    file = download(url)
    
    # Ouvertire et lecture des lignes du fichier
    f = open(file, "r")
    doc = readlines(f)
    
    # Recherche de l'entête des données
    hh = "Station        Date                D\xe9bit (m\xb3/s)   Remarque"
    h = argmax(doc .== hh)
    
    # Récupérations des données sous l'entête
    data = doc[h+1:end];
    data = split.(data)
    data = data[length.(data) .== 4]  # Pour conserver seulement les lignes complètes 
    
    #########################################
    # DONNÉES :
    
    # Station
    station = parse.(Int, getindex.(data, 1));

    # Dates
    d = split.(getindex.(data, 2), "/")
    dates = Date[]
    for i = 1:length(d)
        push!(dates, Date(parse.(Int, d[i])...))
    end

    # Débits
    discharge = parse.(Float64, getindex.(data, 3));

    # Remarques 
    code = String.(getindex.(data, 4))
    #########################################

    # Création du DataFrame
    df = DataFrame(Station = station, Date = dates, Débit = discharge, Remarque = code)
    
    return df
end



# Fonctions utiles pour la Partie 2

## Structures et fonctions (à intégrer dans *HBM.jl*)

import Extremes.showfittedEVA
import Extremes.returnlevel   
import Extremes.cint

## Structures

struct ObsHBM{T} <: fittedEVA{T}
    "pseudo observations"
    pseudoobs::Vector{Vector{Distribution}}
    
    "GEV parameters"
    gevparameters::Extremes.BayesianEVA{T}
    
    "maxima MCMC outputs"
    maxima::Mamba.Chains
end

function showfittedEVA(io::IO, obj::ObsHBM; prefix::String = "")
    println(io, prefix, "ObsHBM")
    println(io, prefix, "gevparameters :")
    showfittedEVA(io, obj.gevparameters, prefix = prefix*"\t")
    println(io)
    println(io, prefix, "maxima :")
    Extremes.showChain(io, obj.maxima, prefix = prefix*"\t")
    println(io)
    println(io, prefix, "pseudoobs :\t\t", typeof(obj.pseudoobs), "[", length(obj.pseudoobs), "]")
    println(io)
end

struct EstimatedMaxima
    years :: Vector{String}
    values :: Vector{Float64}
end

function Base.show(io::IO, obj::EstimatedMaxima)

    println(io, "EstimatedMaxima")
    println(io, "years :\t\t", typeof(obj.years), "[", length(obj.years), "]")
    println(io, "values :\t\t", typeof(obj.values), "[", length(obj.values), "]")
end

## TO-DO : remplacer \mu et \sigma par le vecteur de LogN (pseudoobs)
## TO-DO : vérifier format pour covariable sinon ajouter étape intermédiaire pour avoir le format Extremes.Variable()

# Modèle à 3 params
function obs_bhm2(μh::Matrix{Float64}, σh::Matrix{Float64}; δ₀::Real=0.5,
    warmup::Int=10000, thin::Int=10, niter::Int=20000, adapt::Symbol=:warmup)

    years, S = size(μh)

    Y = zeros(years, niter)
    params = zeros(3, niter)

    #acceptance counts for the Metropolis-Hastings step
    accY = zeros(years)
    accgev = zeros(3)

    #initialization
    Y[:, 1] = mean(exp.(μh), dims = 2)
    fd = Extremes.gevfit(Y[:, 1])
    params[:, 1] = fd.θ̂

    δY = δ₀ * ones(years)
    δ = δ₀ * ones(3)
    uY = rand(years)

    @showprogress for iter=2:niter

        rand!(uY)

        #Updating the maxima

        for y = 1:years

            candidate = max(1e-3, rand(Normal(Y[y, iter-1], δY[y])))
            gev = HBM.getgevparams(params[:, iter-1])

            logpd = sum(logpdf.(LogNormal.(μh[y, :], σh[y , :]), candidate)) -
            sum(logpdf.(LogNormal.(μh[y, :], σh[y , :]), Y[y, iter-1])) +
            #logpd = sum(logpdf.(Normal.(μh[y, :], σh[y , :]), log(candidate))) -
            #sum(logpdf.(Normal.(μh[y, :], σh[y , :]), log(Y[y, iter-1]))) - log(candidate) + log(Y[y, iter-1]) +
            logpdf(GeneralizedExtremeValue(gev...), candidate) -
            logpdf(GeneralizedExtremeValue(gev...), Y[y, iter-1])

            if logpd > log(uY[y])
                Y[y, iter] = candidate
                accY[y] += 1
            else
                Y[y, iter] = Y[y, iter-1]
            end
        end

        data_layer = BlockMaxima(Variable("y", Y[:, iter]))

        #Updating the GEV parameters

        params[:, iter] = params[:, iter - 1]

        new_η = rand(Normal(params[1, iter], δ[1]))
        new = vcat(new_η, params[2:end, iter])
        logpd = Extremes.loglike(data_layer, new) - Extremes.loglike(data_layer, params[:, iter])

        if logpd > log(rand())
            params[1, iter] = new_η
            accgev[1] += 1
        end

        new_ζ = rand(Normal(params[2, iter], δ[2]))
        new = vcat(params[1, iter], new_ζ, params[3, iter])
        logpd = Extremes.loglike(data_layer, new) - Extremes.loglike(data_layer, params[:, iter])

        if logpd > log(rand())
            params[2, iter] = new_ζ
            accgev[2] += 1
        end

        new_ξ = rand(Normal(params[3, iter], δ[3]))
        new = vcat(params[1:2, iter], new_ξ)
        logpd = Extremes.loglike(data_layer, new) - Extremes.loglike(data_layer, params[:, iter]) +
        logpdf(Beta(6., 9.), new_ξ + 0.5) - logpdf(Beta(6., 9.), params[3, iter] + 0.5)

        if logpd > log(rand())
            params[3, iter] = new_ξ
            accgev[3] += 1
        end

        # Updating the stepsize
        if iter % 50 == 0
            if !(adapt == :none)
                if (iter <= warmup) | (adapt==:all)
                    accrate = accY ./ 50
                    δY = HBM.update_stepsize.(δY, accrate)
                    accY = zeros(years)
                    accrate = accgev ./ 50
                    δ = HBM.update_stepsize.(δ, accrate)
                    accgev = zeros(3)
                end
            end
        end
    end

    #Extracting output
    parmnames = ["η", "ζ", "ξ"]
    res_params = Mamba.Chains(collect(params'), names=parmnames)
    res_params = res_params[warmup+1:thin:niter, :, :]
    
    maxnames = ["Y[$y]" for y = 1:years]
    res_maxima = Mamba.Chains(collect(Y'), names=maxnames)
    res_maxima = res_maxima[warmup+1:thin:niter, :, :]
    
    return res_params, res_maxima
end

# Modèle à 4 params
function obs_bhm2(μh::Matrix{Float64}, σh::Matrix{Float64}, locationcov::Vector{<:DataItem}; δ₀::Real=0.5,
    warmup::Int=10000, thin::Int=10, niter::Int=20000, adapt::Symbol=:warmup)

    years, S = size(μh)

    Y = zeros(years, niter)
    params = zeros(4, niter)

    #acceptance counts for the Metropolis-Hastings step
    accY = zeros(years)
    accgev = zeros(4)

    #initialization
    Y[:, 1] = mean(exp.(μh), dims = 2)
    data_layer = BlockMaxima(Variable("y", Y[:, 1]), locationcov = locationcov)
    fd = Extremes.fit(data_layer)
    params[:, 1] = fd.θ̂
    δY = δ₀ * ones(years)
    δ = δ₀ * ones(4)
    uY = rand(years)

    @showprogress for iter=2:niter

        rand!(uY)

        #Updating the maxima

        for y = 1:years

            candidate = max(1e-3, rand(Normal(Y[y, iter-1], δY[y])))
            gev = getgevparams(params[:, iter-1], x=locationcov[1].value[y])

            logpd = sum(logpdf.(LogNormal.(μh[y, :], σh[y , :]), candidate)) -
            sum(logpdf.(LogNormal.(μh[y, :], σh[y , :]), Y[y, iter-1])) +
            #logpd = sum(logpdf.(Normal.(μh[y, :], σh[y , :]), log(candidate))) -
            #sum(logpdf.(Normal.(μh[y, :], σh[y , :]), log(Y[y, iter-1]))) - log(candidate) + log(Y[y, iter-1]) +
            logpdf(GeneralizedExtremeValue(gev...), candidate) -
            logpdf(GeneralizedExtremeValue(gev...), Y[y, iter-1])

            if logpd > log(uY[y])
                Y[y, iter] = candidate
                accY[y] += 1
            else
                Y[y, iter] = Y[y, iter-1]
            end
        end

        data_layer = BlockMaxima(Variable("y", Y[:, iter]), locationcov = locationcov)

        #Updating the GEV parameters

        params[:, iter] = params[:, iter - 1]

        new_η0 = rand(Normal(params[1, iter], δ[1]))
        new = vcat(new_η0, params[2:end, iter])
        logpd = Extremes.loglike(data_layer, new) - Extremes.loglike(data_layer, params[:, iter])

        if logpd > log(rand())
            params[1, iter] = new_η0
            accgev[1] += 1
        end

        new_η1 = rand(Normal(params[2, iter], δ[2]))
        new = vcat(params[1, iter], new_η1, params[3:end, iter])
        logpd = Extremes.loglike(data_layer, new) - Extremes.loglike(data_layer, params[:, iter])

        if logpd > log(rand())
            params[2, iter] = new_η1
            accgev[2] += 1
        end

        new_ζ = rand(Normal(params[3, iter], δ[3]))
        new = vcat(params[1:2, iter], new_ζ, params[end, iter])
        logpd = Extremes.loglike(data_layer, new) - Extremes.loglike(data_layer, params[:, iter])

        if logpd > log(rand())
            params[3, iter] = new_ζ
            accgev[3] += 1
        end

        new_ξ = rand(Normal(params[4, iter], δ[4]))
        new = vcat(params[1:3, iter], new_ξ)
        logpd = Extremes.loglike(data_layer, new) - Extremes.loglike(data_layer, params[:, iter]) +
        logpdf(Beta(6., 9.), new_ξ + 0.5) - logpdf(Beta(6., 9.), params[4, iter] + 0.5)

        if logpd > log(rand())
            params[4, iter] = new_ξ
            accgev[4] += 1
        end

        # Updating the stepsize
        if iter % 50 == 0
            if !(adapt == :none)
                if (iter <= warmup) | (adapt==:all)
                    accrate = accY ./ 50
                    δY = HBM.update_stepsize.(δY, accrate)
                    accY = zeros(years)
                    accrate = accgev ./ 50
                    δ = HBM.update_stepsize.(δ, accrate)
                    accgev = zeros(4)
                end
            end
        end
    end

#     #Extracting output
#     parmnames = vcat(["η₀", "η₁", "ζ", "ξ"], ["Y[$y]" for y = 1:years])
#     res = vcat(params, Y)

#     res = Mamba.Chains(collect(res'), names=parmnames)
#     res = res[warmup+1:thin:niter, :, :]

#     return res
    
    #Extracting output
    parmnames = ["η₀", "η₁", "ζ", "ξ"]
    res_params = Mamba.Chains(collect(params'), names=parmnames)
    res_params = res_params[warmup+1:thin:niter, :, :]
    
    maxnames = ["Y[$y]" for y = 1:years]
    res_maxima = Mamba.Chains(collect(Y'), names=maxnames)
    res_maxima = res_maxima[warmup+1:thin:niter, :, :]
    
    return res_params, res_maxima
end


function obsHBMfit(μ::Matrix{Float64}, σ::Matrix{Float64}; 
        locationcov::Vector{<:DataItem} = Vector{Variable}(),
        δ₀::Real=0.5, 
        niter::Int=20000, warmup::Int=10000, 
        thin::Int=10, adapt::Symbol=:warmup)::ObsHBM
    
    pseudoobs = Vector{Distribution}[]
    for S = 1:6  
        push!(pseudoobs, LogNormal.(μ[:,S], σ[:,S]))
    end

    locationcovstd = Extremes.standardize.(locationcov)
    
    if isempty(locationcov)
        gevparams, maxima = obs_bhm2(μ, σ, δ₀=δ₀, niter=niter, warmup=warmup, thin=thin, adapt=adapt);
        model = Extremes.BlockMaxima(Variable("PseudoObs", Float64[]))
    else
        gevparams, maxima = obs_bhm2(μ, σ, locationcov, δ₀=δ₀, niter=niter, warmup=warmup, thin=thin, adapt=adapt);
    
        new_locationcovstd = similar(locationcovstd)
        for i = 1:length(locationcovstd)
            new_locationcovstd[i] = VariableStd(locationcovstd[i].name, Float64[], NaN, NaN)
        end
        
        model = Extremes.BlockMaxima(Variable("PseudoObs", Float64[]), locationcov=new_locationcovstd)
    end
    
    return ObsHBM(pseudoobs, BayesianEVA(model, gevparams), maxima)

end


function getgevparams(vec::Array{Float64}; x::Float64=0.)
    n = length(vec)
    if n == 3
        return vec[1], exp(vec[2]), vec[3]
    elseif n == 4
        return vec[1] + x * vec[2], exp(vec[3]), vec[4]
    elseif n == 5
        return vec[1] + x * vec[2], exp(vec[3] + x * vec[4]), vec[5]
    end
end


function returnlevel(fm::ObsHBM{BlockMaxima}, returnPeriod::Real)::ReturnLevel              
                                                                                            
    @assert returnPeriod > zero(returnPeriod) "the return period should be positive."     
                                                                                          
    # quantile level                                                                      
    p = 1-1/returnPeriod                                                                  
                                                                                          
    Q = Extremes.quantile(fm.gevparameters, p)                                                                   
                                                                                          
    return Extremes.ReturnLevel(Extremes.BlockMaximaModel(fm), returnPeriod, Q)           
                                                                                          
end 

function cint(rl::ReturnLevel{ObsHBM{BlockMaxima}}, confidencelevel::Real=.95)::Vector{Vector{Real}}
                                                                                            
    @assert rl.returnperiod > zero(rl.returnperiod) "the return period should be positive."
    @assert zero(confidencelevel)<confidencelevel<one(confidencelevel) "the confidence level should be in (0,1)."
                                                                                          
    α = (1 - confidencelevel)                                                             
                                                                                          
    Q = Chains(rl.value)                                                                  
                                                                                          
    ci = Mamba.hpd(Q, alpha = α)                                                          
                                                                                          
    return Extremes.slicematrix(ci.value[:,:,1], dims=2)                                  
                                                                                          
end 

function getmaxima(fm::ObsHBM)
    
    v = Extremes.slicematrix(fm.maxima.value[:,:,1], dims=1)

    values = mean.(v)
    years = fm.maxima.names  # TODO : remplacer par les années plutôt que les noms
    
    return EstimatedMaxima(years, values)
end  
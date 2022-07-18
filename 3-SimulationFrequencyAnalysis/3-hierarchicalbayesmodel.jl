struct HierarchicalBayesModel
    data::Vector{BlockMaxima}
    hyperprior::Vector{<:ContinuousUnivariateDistribution}
end

function HierarchicalBayesModel(data::Vector{BlockMaxima};
    hyperprior::Vector{<:ContinuousUnivariateDistribution} = ContinuousUnivariateDistribution[])
    
    if isempty(hyperprior)
        hyperprior =  Vector{ContinuousUnivariateDistribution}(undef, 2)
        hyperprior[1:2] .= Flat()
    else
        #validateprior(prior, p)
    end
    
    
    return HierarchicalBayesModel(data, hyperprior)
end 

function showhierarchicalbayesmodel(io::IO, obj::HierarchicalBayesModel; prefix::String = "")

    println(io, prefix, "HierarchicalBayesModel")
    println(io, prefix, "  data:\t\t", typeof(obj.data), "[", length(obj.data), "]")
    println(io, prefix, "  hyperparameters: ν, τ")
    println(io, prefix, "  hyperprior: ", "[", obj.hyperprior[1] , [string(", ", obj.hyperprior[k]) for k in 2:length(obj.hyperprior)]...  ,"]")

end


"""
    Base.show(io::IO, obj::HierarchicalBayesModel)
Override of the show function for the objects of type HierarchicalBayesModel.
"""
function Base.show(io::IO, obj::HierarchicalBayesModel)

    showhierarchicalbayesmodel(io, obj)

end
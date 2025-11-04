using Pkg
# Pkg.activate(".")
# Pkg.instantiate()
using Revise
using EasyHybrid
using Lux
using Optimisers
# using GLMakie
using Random
using LuxCore
using CSV, DataFrames
using EasyHybrid.MLUtils
using Statistics
using Plots
using JLD2
using CairoMakie
using OhMyThreads

# 04 - hybrid
testid = "04a_hybridBD";
results_dir = joinpath(@__DIR__, "eval");
target_names = [:BD, :SOCconc, :CF, :SOCdensity];

# input
df = CSV.read(joinpath(@__DIR__, "data/lucas_preprocessed_v20251103.csv"), DataFrame; normalizenames=true)
df = dropmissing(df, target_names);

# scales
scalers = Dict(
    :SOCconc   => 0.158, # log(x)*0.158
    :CF        => 2.2,
    :BD        => 0.52,
    :SOCdensity => 0.165, # log(x)*0.165
);

for tgt in target_names
    # println(tgt, "------------")
    # println(minimum(df[:,tgt]), "  ", maximum(df[:,tgt]))
    if tgt in (:SOCdensity, :SOCconc)
        df[!, tgt] .= log.(df[!, tgt])
        # println(minimum(df[:,tgt]), "  ", maximum(df[:,tgt]))
    end
    df[!, tgt] .= df[!, tgt] .* scalers[tgt]
    # println(minimum(df[:,tgt]), "  ", maximum(df[:,tgt]))
end

# mechanistic model
function SOCD_model(; SOCconc, CF, oBD, mBD)
    soct = exp.(SOCconc ./ scalers[:SOCconc]) ./ 1000 # to fraction
    cft = CF ./ scalers[:CF]   # back to fraction
    BD = (oBD .* mBD) ./ (1.724f0 .* soct .* mBD .+ (1f0 .- 1.724f0 .* soct) .* oBD)
    SOCdensity = soct .*1000 .* BD .* (1 .- cft) # kg/m3
    
    SOCdensity = log.(SOCdensity) .* scalers[:SOCdensity]  # scale to ~[0,1]
    BD = BD .* scalers[:BD]  # scale to ~[0,1]
    return (; BD, SOCconc, CF, SOCdensity, oBD, mBD)  # supervise both BD and SOCconc
end

# param bounds
parameters = (
    SOCconc = (0.01f0, 0.0f0, 1.0f0),   # fraction
    CF      = (0.15f0, 0.0f0, 1.0f0),   # fraction,
    oBD     = (0.20f0, 0.05f0, 0.40f0),  # g/cm^3
    mBD     = (1.20f0, 0.75f0, 2.0f0),  # global
)

# define param for hybrid model
neural_param_names = [:SOCconc, :CF, :mBD]
global_param_names = [:oBD]
forcing = Symbol[]
targets = [:BD, :SOCconc, :SOCdensity, :CF]  # SOCconc is both a param and a target

# predictor
predictors = Symbol.(names(df))[19:end-2]; # CHECK EVERY TIME 
nf = length(predictors)

# search space
mspec = ModelSpec(
    hyper_model = (
        hidden_layers = [
            (256, 128, 64, 32, 16),
            (256, 128, 64, 32),
            (256, 128, 64),
            (256, 128),
            (128, 64, 32, 16),
            (128, 64, 32),
            (128, 64),
            (64, 32, 16),
            (64, 32)
        ],
        activation = [relu, tanh, swish, gelu],
    ),
    hyper_train = (
        batchsize = [64, 128, 256, 512],
        opt = [AdamW(1e-2), AdamW(1e-3), AdamW(1e-4)],
    )
)


# cross-validation
k = 5;
folds = make_folds(df, k = k, shuffle = true);

@time @tasks for test_fold in 1:k
    @info "Split data outside of train function. Training fold $val_fold of $k"

    train_folds = setdiff(1:5, test_fold)

    train_idx = findall(in(train_folds), folds)
    test_idx = findall(==(test_fold), folds)

    (x_test, y_test) = prepare_data(hm, df[test_idx, :])


    ho = @thyperopt for i = nhyper
        println("Hyperparameter tuning run $i of $nhyper")

        out = EasyHybrid.tune(
            hybrid_model,  
            df[train_idx, :],      
            mspec;   # hyper param space
            nepochs = 100,
            patience = 20,
            plotting = false,
            show_progress = false,
            random_seed = 42,
            return_model = :best,
        )

        out.best_loss      # the objective the optimizer minimizes
    end

    best_hyperp = best_hyperparams(ho)
    final_model = EasyHybrid.tune(hm_local, df[train_idx, :], mspec;
                                  best_hyperp..., nepochs=200)

    ps, st = final_model.ps, final_model.st
    (x_test, y_test) = prepare_data(hybrid_model, df_test)

    ps, st = best_result.ps, best_result.st
    ŷ_test, st_test = best_hm(x_test, ps, LuxCore.testmode(st))

    ŷ_df = toDataFrame(ŷ_test, targets)
    for tgt in target_names
        df[test_idx, "pred_$(tgt)"] = ŷ_df[:, tgt] 
    end
    param_names = [:oBD, :mBD]
    for p in param_names
        if hasproperty(ŷ_test, p)
            df[test_idx, Symbol("fitted_", p)] = getproperty(ŷ_test, p)
        end
    end
end

CSV.write(joinpath(results_dir, "$(testid)_cv.pred.csv"), df)

# load predictions
jld = joinpath(results_dir, "$(testid)_best_model.jld2")
@assert isfile(jld) "Missing $(jld). Did you train & save best model for $(tname)?"
@load jld val_obs_pred meta
# split output table
val_tables = Dict{Symbol,Vector{Float64}}()
for t in targets
    # expected: t (true), t_pred (pred), and maybe :index if the framework saved it
    have_pred = Symbol(t, :_pred)
    req = Set((t, have_pred))
    @assert issubset(req, Symbol.(names(val_obs_pred))) "val_obs_pred missing $(collect(req)) for $(t). Columns: $(names(val_obs_pred))"
    val_tables[t] = val_obs_pred[:, t]./ scalers[t]
    val_tables[have_pred] = val_obs_pred[:, have_pred]./ scalers[t]
    if t in (:SOCdensity, :SOCconc)
        val_tables[Symbol("$(t)_pred")] = exp.(val_tables[Symbol("$(t)_pred")]) ./ 1000
        val_tables[t] = exp.(val_tables[t]) ./ 1000
    end
end


# helper for metrics calculation
r2_mse(y_true, y_pred) = begin
    ss_res = sum((y_true .- y_pred).^2)
    ss_tot = sum((y_true .- mean(y_true)).^2)
    r2  = 1 - ss_res / ss_tot
    mse = mean((y_true .- y_pred).^2)
    (r2, mse)
end

# accuracy plots for SOCconc, BD, CF in original space
for tname in targets
    y_val_true = val_tables[tname]
    y_val_pred = val_tables[Symbol("$(tname)_pred")]

    # @assert all(in(Symbol.(names(df_out))).([tname, Symbol("$(tname)_pred")])) "Expected columns $(tname) and $(tname)_pred in saved val table."

    r2, mse = r2_mse(y_val_true, y_val_pred)

    plt = histogram2d(
        y_val_pred, y_val_true;
        nbins=(40, 40), cbar=true, xlab="Predicted", ylab="Observed",
        title = string(tname, "\nR²=", round(r2, digits=3), ", MSE=", round(mse, digits=3)),
        normalize=false
    )
    lims = extrema(vcat(y_val_true, y_val_pred))
    Plots.plot!(plt, [lims[1], lims[2]], [lims[1], lims[2]];
        color=:black, linewidth=2, label="1:1 line",
        aspect_ratio=:equal, xlims=lims, ylims=lims
    )
    savefig(plt, joinpath(results_dir, "$(testid)_accuracy_$(tname).png"))
end

# BD vs SOCconc predictions
plt = histogram2d(
    val_tables[:BD_pred], val_tables[:SOCconc_pred];
    nbins      = (30, 30),
    cbar       = true,
    xlab       = "BD",
    ylab       = "SOCconc",
    color      = cgrad(:bamako, rev=true),
    normalize  = false,
    size = (460, 400),
    xlims     = (0, 1.8),
    ylims     = (0, 0.6)
)   
savefig(plt, joinpath(results_dir, "$(testid)_BD.vs.SOCconc.png"));

# save / print parameters: mBD and per-sample oBD
# oBD global
@load jld oBD_physical
@info "Global oBD ≈ $(round(oBD_physical, digits=4))"

@load jld mBD_phys
histogram(mBD_phys; bins=:sturges, xlabel="learned mBD", ylabel="count",
          title="Distribution of learned mBD", legend=false)
vline!([mean(mBD_phys)]; lw=2, label=false)  # mean marker
@info "Saved histogram to $(joinpath(results_dir, "mBD_histogram.png"))"

# # MTD SOCdensity
# socdensity_pred = val_tables[:SOCconc_pred] .* val_tables[:BD_pred] .* (1 .- val_tables[:CF_pred]);
# socdensity_true = val_tables[:SOCdensity];
# r2_sd, mse_sd = r2_mse(socdensity_true, socdensity_pred);
# plt = histogram2d(
#     socdensity_pred, socdensity_true;
#     nbins=(40,40), cbar=true, xlab="Pred SOCdensity MTD", ylab="True SOCdensity",
#     title = "SOCdensity\nR²=$(round(r2_sd,digits=3)), MSE=$(round(mse_sd,digits=3))",
#     normalize=false
# )
# lims = extrema(vcat(socdensity_true, socdensity_pred))
# Plots.plot!(plt, [lims[1], lims[2]], [lims[1], lims[2]];
#     color=:black, linewidth=2, label="1:1 line",
#     aspect_ratio=:equal, xlims=lims, ylims=lims
# )
# savefig(plt, joinpath(results_dir, "$(testid)_accuracy_SOCdensity.MTD.png"));



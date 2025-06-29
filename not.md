# not.md

`not` method added to margins so that this useful function of a bool
can me handled by ForwardDiff.

```julia
using StatsModels

# Suppose df has columns :y, :socio (Bool), :x, :z
fx = @formula(y ~ socio + x + z & not(socio))

# Fits exactly the model y ~ socio + x + z + not1(socio) + z*not1(socio)
m = fit(LinearModel, fx, df)

# And margins(...) will correctly pick up the interaction with !socio
ame_z = margins(m, :z, df)
```

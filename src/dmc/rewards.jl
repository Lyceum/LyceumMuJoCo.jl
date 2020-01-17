const _DEFAULT_VALUE_AT_MARGIN = 0.1

function tolerance(
    x;
    bounds = (0, 0),
    margin = 0,
    sigmoid = quadratic,
    value_at_margin = _DEFAULT_VALUE_AT_MARGIN,
)
    lo, up = bounds
    lo <= up || throw(DomainError("Lower bound must be <= upper bound"))
    margin >= 0 || throw(DomainError("`margin` must be > 0"))
    _tolerance(x, lo, up, margin, sigmoid, value_at_margin)
end

function _tolerance(x::Real, lo, up, margin, sigmoid, vmargin)
    in_bounds = lo <= x <= up
    if iszero(margin)
        return ifelse(in_bounds, one(x), zero(x))
    else
        d = ifelse(x < lo, lo - x, x - up) / margin
        value = in_bounds ? one(x) :  sigmoid(d, vmargin)
        return convert(typeof(x), value)
    end
end

function _tolerance(xs::AbstractArray, args...)
    map(x -> _tolerance(x, args...), xs)
end


function quadratic(x::Number, value_at_1::Number)
    scale = sqrt(1 - value_at_1)
    scaled_x = scale * x
    ifelse(abs(scaled_x) < 1, 1 - scaled_x ^ 2, zero(typeof(x)))
end
quadratic(xs::AbstractArray, value_at_1::Number) = map(x -> quadratic(x, value_at_1), xs)
quadratic!(xs::AbstractArray, value_at_1::Number) = xs .= quadratic.(xs, value_at_1)


function _check_value_at_1(x::Number)
    0 < x < 1 || throw(DomainError("`value_at_1` must be in range [0, 1)"))
end
function _check_value_at_1_strict(x::Number)
    0 < x < 1 || throw(DomainError("`value_at_1` must be in range (0, 1)"))
end

"""
Return (invlink, dinvlink, d2invlink) for model `m`.
"""
function link_functions(model)
    link = model.resp.link
    return (
      invlink   = η -> linkinv(link, η),
      dinvlink  = η -> mueta(link,  η),
      d2invlink = η -> mueta2(link, η),
    )
end
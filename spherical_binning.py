from math import pi

def _deg_to_rad(degs):
  return [deg*pi/180 for deg in degs]

ALPHA_BINS_DEG = [-150., -120., -90., -60., -30., 0.,
                   30., 60., 90., 120., 150., 180.]
THETA_BINS_DEG = [15., 45., 75., 105., 135., 165., 180.]

ALPHA_BINS_RAD = _deg_to_rad(ALPHA_BINS_DEG)
THETA_BINS_RAD = _deg_to_rad(THETA_BINS_DEG)

def get_bins(alpha, theta):
  ''' Returns the bin boundaries for alpha plus the bin boundaries for the two
      neighbouring bins. Similarly for theta
  '''
  alpha_c_ix, theta_c_ix = _get_central_bin(alpha, theta)

  alpha_bins, theta_bins = _get_surrounding_bins(alpha_c_ix, theta_c_ix)

  return alpha_bins, theta_bins 

def embed_index_from_bin(abin, tbin):
  ''' Given the alpha bounds and the theta bounds calculate
      the position in the 84-d embedding matrix.

      Alpha bins are numbered 0 to 11, theta bins 0 to 6.

      Given the alpah and theata bin numbers the embedding index
      is calculated as:

         <alpha bin no.> * 7 + <theta bin no.>
  '''
  alpha_bin_no = _get_bin_number_from_u_bound(abin[1], ALPHA_BINS_RAD)
  theta_bin_no = _get_bin_number_from_u_bound(tbin[1], THETA_BINS_RAD)

  embed_ix = alpha_bin_no * 7 + theta_bin_no
  assert embed_ix >=0
  assert embed_ix < 84, f"embed_ix ({embed_ix} - {alpha_bin_no}, {theta_bin_no}) must be less than 84"

  return embed_ix

def _get_bin_number_from_u_bound(bound, bound_list):
  for ix,el in enumerate(bound_list):
    if bound <= el:
      return ix
  # bound is greater than maximum in list
  return -1

def _get_surrounding_bins(abin, tbin):
  abin_ixs, tbin_ixs = _get_surrounding_bins_index(abin, tbin)

  abins = ALPHA_BINS_DEG
  get_alpha_bin_limits = lambda ix: (abins[ix-1], abins[ix]) if ix > 0 else (-180., -150.)
  alpha_bin_boundaries = [get_alpha_bin_limits(aix) for aix in abin_ixs]

  tbins = THETA_BINS_DEG
  get_theta_bin_limits = lambda ix: (tbins[ix-1], tbins[ix]) if ix > 0 else (0., 15.)
  theta_bin_boundaries = [get_theta_bin_limits(tix)  for tix in tbin_ixs]

  # convert to radians
  alpha_bin_boundaries = [_deg_to_rad(l_u) for l_u in alpha_bin_boundaries]
  theta_bin_boundaries = [_deg_to_rad(l_u) for l_u in theta_bin_boundaries]

  return alpha_bin_boundaries, theta_bin_boundaries

def _get_surrounding_bins_index(abin, tbin):
  # calculate the neighbours of alpha and theta bins
  abin_l = abin-1 if abin-1 >= 0  else 11
  abin_u = abin+1 if abin+1 <= 11 else  0
  tbin_l = tbin-1 if tbin-1 >= 0  else  1
  tbin_u = tbin+1 if tbin+1 <= 6  else  5

  return [abin_l, abin, abin_u], [tbin_l, tbin, tbin_u]

def _get_central_bin(alpha, theta):
  assert(-pi <= alpha and alpha <= pi)
  assert(-pi <= theta and theta <= pi)

  abin = sum([alpha > bound 
              for bound in _deg_to_rad(ALPHA_BINS_DEG)] 
  )

  tbin = sum([abs(theta) > bound 
                  for bound in _deg_to_rad(THETA_BINS_DEG)]

  )

  return abin, tbin



if __name__ == '__main__':
  print('''
  #
  # Test _get_central_bins
  #
  ''')
  alpha = 93*pi/180 
  theta = -63*pi/180
  abin, tbin = _get_central_bin(alpha, theta)

  print(f'for alpha={alpha}, theta={theta} _get_central_bin returns {abin}, {tbin}')

  print('''
  #
  # Test _get_surrounding_bins_index
  #
  ''')
  abin = 2
  tbin = 3
  print(f'for abin={abin}, tbin={tbin} _get_surrounding_bins_index => {_get_surrounding_bins_index(abin, tbin)}')

  abin = 0
  tbin = 6
  print(f'for abin={abin}, tbin={tbin} _get_surrounding_bins_index => {_get_surrounding_bins_index(abin, tbin)}')

  abin = 11
  tbin = 0
  print(f'for abin={abin}, tbin={tbin} _get_surrounding_bins_index => {_get_surrounding_bins_index(abin, tbin)}')

  print('''
  #
  # Test _get_surronding_bins
  #
  ''')
  abin = 2
  tbin = 3
  _get_surrounding_bins(abin, tbin)

  print('''
  #
  #  Test get_bins
  #
  ''')
  alpha = 93*pi/180 
  theta = -63*pi/180
  print(f'for alpha={alpha}, theta={theta} get_bins => {get_bins(alpha, theta)}')

  print('')
  alpha = 179*pi/180 
  theta = -179*pi/180
  print(f'for alpha={alpha}, theta={theta} get_bins => {get_bins(alpha, theta)}')

 

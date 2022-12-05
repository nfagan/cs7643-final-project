function p = proj_root()
p = fileparts( fileparts(which(mfilename)) );
end
DIRECTORY=/work/grana_maxillo/Mamba3DMedModels/
chgrp -R grana_maxillo $DIRECTORY
setfacl -dR -m u::rwx,g::rwx $DIRECTORY
chmod -R g+rwxs $DIRECTORY

<?
#
# constants.m :: Contains the model constants / parameters
#
# This file is part of Myokit
#  Copyright 2011-2017 Maastricht University
#  Licensed under the GNU General Public License v3.0
#  See: http://myokit.org
#
# Authors:
#  Michael Clerx
#
?>%
% Constants for <?= model.name() ?>
%
function c = constants()
<?
for label, eq_list in equations.iteritems():
    print('')
    print('% ' + label)
    for eq in eq_list.equations(const=True):
        var = eq.lhs.var()
        if 'desc' in var.meta:
            print('% ' + '% '.join(str(var.meta['desc']).splitlines()))
        print(e(eq) + ';')
?>
end

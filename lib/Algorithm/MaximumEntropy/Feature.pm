package Algorithm::MaximumEntropy::Feature;

use Mouse;

has 'label' => (
    is => 'rw',
    isa => 'Str'
    );

has 'vector' => (
    is => 'rw',
    isa => 'ArrayRef'
    );

sub BUILD {
    
}

__PACKAGE__->meta->make_immutable();

1;
